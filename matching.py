"""
Turns a table of pairwise similarity features into match / no-match
predictions.

Two scoring modes are supported:

1. "rule" (default, no training data required): a Fellegi-Sunter-style
   weighted sum of the similarity features, squashed through a sigmoid into
   a [0, 1] match score. Weights are hand-set to favour precision, which is
   what an F0.5 metric rewards. This is what `run_predict.py` uses out of
   the box, since only the (unlabeled) test sources are available.

2. "classifier": a scikit-learn classifier (LogisticRegression,
   GradientBoostingClassifier, ...) trained on the same feature columns via
   `train_classifier.py`, once labeled training data (train_source*.tsv +
   train_ground_truth.tsv) is available. Loaded with joblib and used in
   place of the rule-based scorer -- same interface, same downstream
   thresholding/selection logic.
"""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "name_jw", "name_token_jaccard", "addr_jw", "addr_token_jaccard",
    "phonetic_match", "sorted_key_match", "postal_match", "state_match",
    "tfidf_cosine",
]

# Hand-set weights for the no-training-data-required rule-based scorer.
#
# `phonetic_match` and `sorted_key_match` are deliberately given very low
# weight here even though they are exact-match features. They are blocking
# keys first and foremost (built from just the first few name tokens), so
# they are *supposed* to over-match -- e.g. every "Maison de Santé ..." (a
# common French healthcare-facility name prefix) or every "Sharma Traders"
# shares one, regardless of whether the two records are the same business.
# Treating that kind of agreement as strong match evidence was tried first
# and produced exactly that failure mode (see the documentation's error
# analysis); the fix is to lean on address agreement and full-text TF-IDF
# similarity as the primary discriminators, and use the exact-key agreement
# only as a small tie-breaking nudge.
DEFAULT_WEIGHTS = {
    "name_jw": 2.5,
    "name_token_jaccard": 1.0,
    "addr_jw": 2.0,
    "addr_token_jaccard": 2.5,
    "phonetic_match": 0.3,
    "sorted_key_match": 0.5,
    "postal_match": 2.0,
    "state_match": 0.5,
    "tfidf_cosine": 2.0,
    "bias": -5.5,
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def score_pairs_rule_based(pairs: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """Adds a `match_score` column in [0, 1] using the weighted-sum scorer."""
    if len(pairs) == 0:
        pairs = pairs.copy()
        pairs["match_score"] = pd.Series(dtype="float64")
        return pairs

    weights = weights or DEFAULT_WEIGHTS
    pairs = pairs.copy()

    feats = pairs[FEATURE_COLUMNS].fillna(0.0)
    linear = np.full(len(pairs), weights.get("bias", 0.0), dtype="float64")
    for col in FEATURE_COLUMNS:
        linear += weights.get(col, 0.0) * feats[col].to_numpy(dtype="float64")

    pairs["match_score"] = _sigmoid(linear)
    return pairs


def score_pairs_with_classifier(pairs: pd.DataFrame, clf) -> pd.DataFrame:
    """Adds a `match_score` column using a fitted sklearn-compatible classifier
    (must expose predict_proba and have been trained on FEATURE_COLUMNS, in
    that order)."""
    pairs = pairs.copy()
    if len(pairs) == 0:
        pairs["match_score"] = pd.Series(dtype="float64")
        return pairs
    feats = pairs[FEATURE_COLUMNS].fillna(0.0)
    proba = clf.predict_proba(feats)
    # Assume the positive ("match") class is the second column, as sklearn
    # convention dictates for binary classifiers with classes_ = [0, 1].
    pairs["match_score"] = proba[:, 1]
    return pairs


def select_matches(pairs: pd.DataFrame, threshold: float = 0.5,
                    max_matches_per_entity: int = None) -> pd.DataFrame:
    """Keep only candidate pairs whose match_score clears the threshold.
    Optionally cap the number of matches kept per S1 entity (highest score
    first) -- useful if you want to bound false positives per entity."""
    if len(pairs) == 0:
        return pairs
    selected = pairs[pairs["match_score"] >= threshold].copy()
    if max_matches_per_entity is not None and len(selected) > 0:
        selected = (
            selected.sort_values("match_score", ascending=False)
            .groupby("source1_entity_id", group_keys=False)
            .head(max_matches_per_entity)
        )
    return selected


def assemble_results(s1_ids, selected_pairs: pd.DataFrame) -> pd.DataFrame:
    """Build one row per S1 entity_id (matching_results.tsv format), including
    S1 entities with zero predicted matches (empty matched_entity_ids)."""
    if len(selected_pairs) > 0:
        matched_map = (
            selected_pairs.groupby("source1_entity_id")["candidate_entity_id"]
            .agg(lambda ids: ",".join(sorted(ids)))
        )
    else:
        matched_map = pd.Series(dtype="object")

    rows = [
        {"source1_entity_id": sid, "matched_entity_ids": matched_map.get(sid, "")}
        for sid in s1_ids
    ]
    return pd.DataFrame(rows)
