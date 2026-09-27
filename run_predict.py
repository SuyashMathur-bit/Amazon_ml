#!/usr/bin/env python3
"""
Run entity-resolution prediction on the Business Entity Resolution test sources.

Usage
-----
    python src/run_predict.py \\
        --data-dir /path/to/test/dir \\
        --output-dir output \\
        --threshold 0.55

Expects <data-dir>/test_source1.tsv, test_source2.tsv, test_source3.tsv, each
with columns: entity_id, business_name, business_address, country.

Writes:
    <output-dir>/candidate_pairs.tsv   (S1 entity -> blocked candidate ids)
    <output-dir>/matching_results.tsv  (S1 entity -> predicted matched ids)

By default this uses the no-training-required rule-based scorer. Pass
--classifier path/to/model.joblib to use a classifier trained with
train_classifier.py instead.
"""
import argparse
import os
import sys
import time

import jellyfish
import joblib

from io_utils import read_tsv_with_progress
from pipeline import run_pipeline


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True,
                    help="Directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv")
    p.add_argument("--output-dir", default="output",
                    help="Directory to write candidate_pairs.tsv / matching_results.tsv into (default: output)")
    p.add_argument("--source1", default="test_source1.tsv", help="Filename of the Source-1 (reference) file")
    p.add_argument("--source2", default="test_source2.tsv", help="Filename of the Source-2 file")
    p.add_argument("--source3", default="test_source3.tsv", help="Filename of the Source-3 file")
    p.add_argument("--top-k", type=int, default=10,
                    help="Neighbours to keep per S1 entity from TF-IDF blocking, per source (default: 10)")
    p.add_argument("--tfidf-batch-size", type=int, default=500,
                    help="S1 batch size for TF-IDF nearest-neighbour blocking queries (default: 500)")
    p.add_argument("--threshold", type=float, default=0.55,
                    help="match_score cutoff in [0,1]; higher = fewer/more precise matches (default: 0.55)")
    p.add_argument("--max-matches-per-entity", type=int, default=None,
                    help="Optional cap on predicted matches kept per S1 entity (default: no cap)")
    p.add_argument("--classifier", default=None,
                    help="Optional path to a joblib-saved sklearn classifier (see train_classifier.py). "
                         "If omitted, the rule-based scorer in matching.py is used.")
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return p.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet

    s1_path = os.path.join(args.data_dir, args.source1)
    s2_path = os.path.join(args.data_dir, args.source2)
    s3_path = os.path.join(args.data_dir, args.source3)
    for path in (s1_path, s2_path, s3_path):
        if not os.path.exists(path):
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    t0 = time.time()
    if verbose:
        print("Reading Source 1...")
    s1_df = read_tsv_with_progress(s1_path)
    if verbose:
        print("Reading Source 2...")
    s2_df = read_tsv_with_progress(s2_path)
    if verbose:
        print("Reading Source 3...")
    s3_df = read_tsv_with_progress(s3_path)
    if verbose:
        print(f"S1: {len(s1_df):,} | S2: {len(s2_df):,} | S3: {len(s3_df):,}  "
              f"(loaded in {time.time() - t0:.1f}s)\n")

    classifier = None
    if args.classifier:
        if verbose:
            print(f"Loading classifier from {args.classifier} ...")
        classifier = joblib.load(args.classifier)

    t0 = time.time()
    summary = run_pipeline(
        s1_df, s2_df, s3_df,
        output_dir=args.output_dir,
        jellyfish_module=jellyfish,
        top_k=args.top_k,
        tfidf_batch_size=args.tfidf_batch_size,
        threshold=args.threshold,
        max_matches_per_entity=args.max_matches_per_entity,
        classifier=classifier,
        verbose=verbose,
    )
    if verbose:
        print(f"\nTotal pipeline time: {time.time() - t0:.1f}s")
    return summary


if __name__ == "__main__":
    main()
