"""
Turns blocking output (S1 entity -> candidate entity ids) into a table of
pairwise similarity features, one row per (source1_entity_id,
candidate_entity_id) pair. This is the input to the match-scoring step.

All features are cheap, symbolic string-similarity measures -- no external
data, no pretrained embeddings, no network calls -- which keeps the whole
pipeline trivially inside the "no external lookups / small model" constraint
the challenge sets.
"""
import numpy as np
import pandas as pd

NORM_COLS = [
    "entity_id", "business_name", "business_address", "country",
    "_name_core", "_name_canon", "_addr_norm", "_sorted_key",
    "_phonetic_key", "_postal_code", "_state", "_text_for_tfidf",
]


def build_pairs_frame(candidates: dict, s1_part: pd.DataFrame, s2_part: pd.DataFrame,
                       s3_part: pd.DataFrame) -> pd.DataFrame:
    """
    candidates: dict s1_entity_id -> iterable of candidate entity_ids, restricted
    to the S1/S2/S3 records present in this partition (as produced by
    blocking.generate_candidates_for_partition).
    """
    pair_rows = [
        (s1_id, cid)
        for s1_id, cids in candidates.items()
        for cid in cids
    ]
    if not pair_rows:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])

    pairs = pd.DataFrame(pair_rows, columns=["source1_entity_id", "candidate_entity_id"])

    other_part = pd.concat([s2_part[NORM_COLS], s3_part[NORM_COLS]], ignore_index=True)

    s1_lookup = s1_part[NORM_COLS].add_prefix("s1_")
    other_lookup = other_part[NORM_COLS].add_prefix("cand_")

    pairs = pairs.merge(s1_lookup, left_on="source1_entity_id", right_on="s1_entity_id", how="left")
    pairs = pairs.merge(other_lookup, left_on="candidate_entity_id", right_on="cand_entity_id", how="left")
    pairs = pairs.drop(columns=["s1_entity_id", "cand_entity_id"])
    return pairs


def _token_jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 0.0
    union = ta | tb
    if not union:
        return 0.0
    return len(ta & tb) / len(union)


def _exact_match_known(a, b) -> float:
    """1.0 if equal & both non-empty, 0.0 if different, np.nan if either is unknown."""
    if not a or not b:
        return np.nan
    return 1.0 if a == b else 0.0


def _cosine_via_vectorizer(vectorizer, texts_a, texts_b):
    """Cosine similarity between paired texts, using an already-fitted
    TfidfVectorizer (so IDF weights reflect the real corpus, not just the
    pairs being scored). Rows from TfidfVectorizer are L2-normalized, so the
    row-wise elementwise-product sum below *is* the cosine similarity."""
    vecs_a = vectorizer.transform(texts_a)
    vecs_b = vectorizer.transform(texts_b)
    return np.asarray(vecs_a.multiply(vecs_b).sum(axis=1)).ravel()


def add_similarity_features(pairs: pd.DataFrame, jellyfish_module,
                             s2_vectorizer=None, s3_vectorizer=None) -> pd.DataFrame:
    """
    s2_vectorizer / s3_vectorizer: the TfidfVectorizer instances returned by
    blocking.generate_candidates_for_partition for this same partition --
    fit on the *whole* S2 / S3 partition, so cosine similarity here uses
    correct, corpus-wide IDF weights rather than being computed from scratch
    on just the (small, unrepresentative) set of candidate-pair texts.
    """
    if len(pairs) == 0:
        for col in ["name_jw", "name_token_jaccard", "addr_jw", "addr_token_jaccard",
                    "phonetic_match", "sorted_key_match", "postal_match", "state_match",
                    "tfidf_cosine"]:
            pairs[col] = pd.Series(dtype="float64")
        return pairs

    pairs = pairs.copy()

    pairs["name_jw"] = [
        jellyfish_module.jaro_winkler_similarity(a or "", b or "")
        for a, b in zip(pairs["s1__name_core"], pairs["cand__name_core"])
    ]
    pairs["name_token_jaccard"] = [
        _token_jaccard(a or "", b or "")
        for a, b in zip(pairs["s1__name_core"], pairs["cand__name_core"])
    ]
    pairs["addr_jw"] = [
        jellyfish_module.jaro_winkler_similarity(a or "", b or "")
        for a, b in zip(pairs["s1__addr_norm"], pairs["cand__addr_norm"])
    ]
    pairs["addr_token_jaccard"] = [
        _token_jaccard(a or "", b or "")
        for a, b in zip(pairs["s1__addr_norm"], pairs["cand__addr_norm"])
    ]
    pairs["phonetic_match"] = [
        _exact_match_known(a, b)
        for a, b in zip(pairs["s1__phonetic_key"], pairs["cand__phonetic_key"])
    ]
    pairs["sorted_key_match"] = [
        _exact_match_known(a, b)
        for a, b in zip(pairs["s1__sorted_key"], pairs["cand__sorted_key"])
    ]
    pairs["postal_match"] = [
        _exact_match_known(a, b)
        for a, b in zip(pairs["s1__postal_code"], pairs["cand__postal_code"])
    ]
    pairs["state_match"] = [
        _exact_match_known(a, b)
        for a, b in zip(pairs["s1__state"], pairs["cand__state"])
    ]

    # TF-IDF cosine similarity: route each pair to the vectorizer fit on its
    # candidate's own source (S2 or S3), so both sides of the cosine
    # similarity are scored in the same, corpus-correct vector space.
    cosine = np.zeros(len(pairs), dtype="float64")
    is_s2 = pairs["candidate_entity_id"].str.startswith("S2-")
    is_s3 = pairs["candidate_entity_id"].str.startswith("S3-")

    if s2_vectorizer is not None and is_s2.any():
        texts_a = pairs.loc[is_s2, "s1__text_for_tfidf"].fillna("").astype(str)
        texts_b = pairs.loc[is_s2, "cand__text_for_tfidf"].fillna("").astype(str)
        cosine[is_s2.to_numpy()] = _cosine_via_vectorizer(s2_vectorizer, texts_a, texts_b)

    if s3_vectorizer is not None and is_s3.any():
        texts_a = pairs.loc[is_s3, "s1__text_for_tfidf"].fillna("").astype(str)
        texts_b = pairs.loc[is_s3, "cand__text_for_tfidf"].fillna("").astype(str)
        cosine[is_s3.to_numpy()] = _cosine_via_vectorizer(s3_vectorizer, texts_a, texts_b)

    pairs["tfidf_cosine"] = cosine
    return pairs
