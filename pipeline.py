"""
End-to-end prediction pipeline: raw test_source1/2/3.tsv in, candidate_pairs.tsv
and matching_results.tsv out.

Runs one country partition at a time (blocking -> feature engineering ->
scoring -> thresholding -> write-and-discard) so peak memory stays bounded by
the largest single-country partition rather than the whole dataset.
"""
import gc
import os

import blocking
import features as feat
import matching
from io_utils import append_tsv


def run_pipeline(s1_df, s2_df, s3_df, output_dir, jellyfish_module,
                  top_k=10, tfidf_batch_size=500,
                  threshold=0.5, max_matches_per_entity=None, weights=None,
                  classifier=None, verbose=True):
    """
    Parameters
    ----------
    s1_df, s2_df, s3_df : raw dataframes with columns
        [entity_id, business_name, business_address, country]
    output_dir : directory to write candidate_pairs.tsv and matching_results.tsv into
        (files are overwritten at the start of the run, then appended to per partition)
    jellyfish_module : the imported `jellyfish` package (passed explicitly so this
        module has no hard import-time dependency on it)
    threshold : match_score cutoff in [0, 1]; higher = fewer, more precise matches
    max_matches_per_entity : optional cap on predicted matches per S1 entity
    weights : optional override of matching.DEFAULT_WEIGHTS (rule-based scorer only)
    classifier : optional fitted sklearn classifier (see matching.score_pairs_with_classifier);
        if provided, it is used INSTEAD of the rule-based scorer

    Returns
    -------
    dict of summary counters.
    """
    os.makedirs(output_dir, exist_ok=True)
    candidates_path = os.path.join(output_dir, "candidate_pairs.tsv")
    results_path = os.path.join(output_dir, "matching_results.tsv")
    for p in (candidates_path, results_path):
        if os.path.exists(p):
            os.remove(p)

    total_s1 = 0
    total_pairs = 0
    total_matched_s1 = 0

    for country, s1_part, s2_part, s3_part in blocking.iter_country_partitions(
        s1_df, s2_df, s3_df, jellyfish_module, verbose=verbose
    ):
        candidates, s2_vectorizer, s3_vectorizer = blocking.generate_candidates_for_partition(
            s1_part, s2_part, s3_part, top_k=top_k,
            tfidf_batch_size=tfidf_batch_size, verbose=verbose,
        )
        # Make sure every S1 id in this partition has an entry, even with no candidates.
        for s1_id in s1_part["entity_id"]:
            candidates.setdefault(s1_id, set())

        cand_frame = blocking.candidates_to_frame(
            {k: sorted(v) for k, v in candidates.items()}
        )
        append_tsv(cand_frame, candidates_path)
        total_pairs += sum(len(v) for v in candidates.values())

        pairs = feat.build_pairs_frame(candidates, s1_part, s2_part, s3_part)
        pairs = feat.add_similarity_features(
            pairs, jellyfish_module, s2_vectorizer=s2_vectorizer, s3_vectorizer=s3_vectorizer,
        )

        if classifier is not None:
            pairs = matching.score_pairs_with_classifier(pairs, classifier)
        else:
            pairs = matching.score_pairs_rule_based(pairs, weights=weights)

        selected = matching.select_matches(
            pairs, threshold=threshold, max_matches_per_entity=max_matches_per_entity,
        )
        results = matching.assemble_results(s1_part["entity_id"], selected)
        append_tsv(results, results_path)

        total_s1 += len(s1_part)
        total_matched_s1 += (results["matched_entity_ids"] != "").sum()

        if verbose:
            print(f"    -> {len(candidates)} S1 entities, {len(pairs)} candidate pairs, "
                  f"{(results['matched_entity_ids'] != '').sum()} predicted with >=1 match\n")

        del candidates, cand_frame, pairs, selected, results
        gc.collect()

    if verbose:
        print(f"Done. {total_s1} S1 entities processed, {total_pairs} candidate pairs "
              f"generated, {total_matched_s1} S1 entities got >=1 predicted match.")
        print(f"Wrote: {candidates_path}")
        print(f"Wrote: {results_path}")

    return {
        "total_s1": total_s1,
        "total_candidate_pairs": total_pairs,
        "total_matched_s1": total_matched_s1,
        "candidates_path": candidates_path,
        "results_path": results_path,
    }
