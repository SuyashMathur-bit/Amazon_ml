#!/usr/bin/env python3
"""
Optional: train a small scikit-learn classifier on labeled data
(train_source1/2/3.tsv + train_ground_truth.tsv) to replace the rule-based
scorer in matching.py. Not required to produce predictions -- run_predict.py
works with no training data via the rule-based scorer -- but will generally
score higher once real labels are available, since the weights are learned
rather than hand-set.

Trains on the same blocking output that run_predict.py would generate, so
the classifier only ever sees the pairs the blocking stage actually
surfaces (consistent train/serve behaviour).

Usage
-----
    python src/train_classifier.py \\
        --data-dir /path/to/train/dir \\
        --ground-truth /path/to/train/dir/train_ground_truth.tsv \\
        --model-out output/classifier.joblib

Any scikit-learn classifier that is (a) MIT/Apache-2.0-licensed and (b)
trivially under the challenge's parameter-count ceiling qualifies --
GradientBoostingClassifier is the default; swap in LogisticRegression with
--model-type logreg for a simpler, more interpretable option.
"""
import argparse
import os
import sys

import jellyfish
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import blocking
import features as feat
from ground_truth import load_ground_truth
from io_utils import read_tsv_with_progress
from matching import FEATURE_COLUMNS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="Directory with train_source1/2/3.tsv")
    p.add_argument("--ground-truth", required=True, help="Path to train_ground_truth.tsv")
    p.add_argument("--source1", default="train_source1.tsv")
    p.add_argument("--source2", default="train_source2.tsv")
    p.add_argument("--source3", default="train_source3.tsv")
    p.add_argument("--model-out", default="output/classifier.joblib")
    p.add_argument("--model-type", choices=["gboost", "logreg"], default="gboost")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--tfidf-batch-size", type=int, default=500)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    s1_path = os.path.join(args.data_dir, args.source1)
    s2_path = os.path.join(args.data_dir, args.source2)
    s3_path = os.path.join(args.data_dir, args.source3)
    for path in (s1_path, s2_path, s3_path, args.ground_truth):
        if not os.path.exists(path):
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    print("Reading training sources...")
    s1_df = read_tsv_with_progress(s1_path)
    s2_df = read_tsv_with_progress(s2_path)
    s3_df = read_tsv_with_progress(s3_path)
    ground_truth = load_ground_truth(args.ground_truth)
    print(f"S1: {len(s1_df):,} | S2: {len(s2_df):,} | S3: {len(s3_df):,} | GT entries: {len(ground_truth):,}\n")

    all_pairs = []
    for country, s1_part, s2_part, s3_part in blocking.iter_country_partitions(
        s1_df, s2_df, s3_df, jellyfish, verbose=True
    ):
        candidates, s2_vectorizer, s3_vectorizer = blocking.generate_candidates_for_partition(
            s1_part, s2_part, s3_part, top_k=args.top_k,
            tfidf_batch_size=args.tfidf_batch_size, verbose=True,
        )
        pairs = feat.build_pairs_frame(candidates, s1_part, s2_part, s3_part)
        pairs = feat.add_similarity_features(
            pairs, jellyfish, s2_vectorizer=s2_vectorizer, s3_vectorizer=s3_vectorizer,
        )
        if len(pairs) == 0:
            continue
        pairs["label"] = [
            1 if cid in ground_truth.get(s1_id, set()) else 0
            for s1_id, cid in zip(pairs["source1_entity_id"], pairs["candidate_entity_id"])
        ]
        all_pairs.append(pairs[["source1_entity_id", "candidate_entity_id", "label"] + FEATURE_COLUMNS])

    full = pd.concat(all_pairs, ignore_index=True)
    print(f"\nTotal labeled candidate pairs: {len(full):,}  "
          f"(positives: {(full['label'] == 1).sum():,}, "
          f"negatives: {(full['label'] == 0).sum():,})")

    # Split by S1 entity (not by row) so a given S1 entity's pairs don't leak
    # across train/val.
    s1_ids = full["source1_entity_id"].unique()
    train_ids, val_ids = train_test_split(s1_ids, test_size=args.val_frac, random_state=args.seed)
    train_df = full[full["source1_entity_id"].isin(train_ids)]
    val_df = full[full["source1_entity_id"].isin(val_ids)]

    X_train, y_train = train_df[FEATURE_COLUMNS].fillna(0.0), train_df["label"]
    X_val, y_val = val_df[FEATURE_COLUMNS].fillna(0.0), val_df["label"]

    if args.model_type == "gboost":
        clf = GradientBoostingClassifier(random_state=args.seed)
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    print(f"\nTraining {args.model_type} on {len(X_train):,} pairs...")
    clf.fit(X_train, y_train)

    val_proba = clf.predict_proba(X_val)[:, 1]
    for thresh in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        val_pred = (val_proba >= thresh).astype(int)
        tp = ((val_pred == 1) & (y_val == 1)).sum()
        fp = ((val_pred == 1) & (y_val == 0)).sum()
        fn = ((val_pred == 0) & (y_val == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f0_5 = (1.25 * precision * recall / (0.25 * precision + recall)
                if (precision and recall and (0.25 * precision + recall) > 0) else float("nan"))
        print(f"  threshold={thresh:.1f}  precision={precision:.3f}  recall={recall:.3f}  "
              f"pair-level F0.5={f0_5:.3f}")

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    joblib.dump(clf, args.model_out)
    print(f"\nSaved classifier to {args.model_out}")
    print("Run predictions with: python src/run_predict.py ... --classifier "
          f"{args.model_out}")
    print("Then pick a final --threshold for run_predict.py using entity-level "
          "F0.5 via evaluate.py on a held-out split, since pair-level and "
          "entity-level F0.5 are related but not identical.")


if __name__ == "__main__":
    main()
