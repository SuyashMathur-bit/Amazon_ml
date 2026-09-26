# Business Entity Resolution

## Overview

This project performs **Business Entity Resolution (ER)** across three data sources:

* **S1** — Reference/source-1 business records
* **S2** — Source-2 business records
* **S3** — Source-3 business records

The objective is to identify which S2 and S3 records refer to the same real-world business as each S1 record.

Because the datasets contain noisy names and addresses, the project uses a two-stage approach:

1. **Blocking / Candidate Generation**
2. **Candidate Matching**

The blocking stage reduces the number of records that need to be compared while maintaining high recall.

---

## Pipeline

```text
Raw S1 / S2 / S3 Data
          │
          ▼
     Normalization
          │
          ├── Country
          ├── Business Name
          └── Address
          │
          ▼
   Country Partitioning
          │
          ▼
      Blocking
          │
    ┌─────┼─────┐
    │     │     │
    ▼     ▼     ▼
Phonetic Token  TF-IDF
    │     │     │
    └─────┼─────┘
          ▼
   Candidate Pairs
          │
          ▼
  Blocking Recall Check
          │
          ▼
   Candidate Matching
          │
          ▼
    Final Predictions
```

---

## 1. Data Normalization

Before blocking, the records are normalized to make different representations of the same business easier to compare.

The pipeline generates normalized fields such as:

```text
_country_norm
_name_core
_name_canon
_addr_norm
_sorted_key
_phonetic_key
_text_for_tfidf
```

Normalization includes:

* Unicode/text normalization
* Business-name normalization
* Legal suffix handling
* Address normalization
* Country normalization
* Token sorting
* Phonetic representation

---

## 2. Country Partitioning

Records are divided into country-specific partitions before candidate generation.

For example:

```text
India
 ├── S1
 ├── S2
 └── S3

USA
 ├── S1
 ├── S2
 └── S3
```

This prevents unnecessary comparisons between records from different countries.

The implementation does not rely on a fixed list of countries, allowing new countries to appear in the test data.

---

## 3. Blocking

Blocking is used to generate a smaller set of possible matches for every S1 record.

### Phonetic Blocking

Business names are converted into phonetic keys using `jellyfish`.

This helps identify names with similar pronunciation despite spelling differences.

### Sorted-Token Blocking

Name tokens are sorted to reduce the effect of word-order differences.

For example:

```text
ABC Global Technologies
```

and

```text
Global Technologies ABC
```

can produce the same token-based key.

### TF-IDF Blocking

Character-level TF-IDF is used to identify records with similar text.

The current implementation uses:

```python
TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 4)
)
```

Nearest-neighbor search is then used to retrieve the top candidate records.

---

## 4. Candidate Generation

The main functions are:

```python
generate_all_candidates()
generate_candidates_for_partition()
```

The workflow is:

```text
generate_all_candidates()
        │
        ▼
Country partition
        │
        ▼
generate_candidates_for_partition()
        │
        ├── S2 phonetic
        ├── S2 token
        ├── S2 TF-IDF
        │
        ├── S3 phonetic
        ├── S3 token
        └── S3 TF-IDF
        │
        ▼
Union of candidates
```

The output is a dictionary such as:

```text
S1-001 → [S2-101, S2-205, S3-301]
S1-002 → [S3-412]
S1-003 → []
```

These are **candidate matches**, not final predictions.

---

## 5. Blocking Recall

After candidate generation, the candidate set is compared with the validation ground truth.

```python
candidates = generate_all_candidates(
    s1_val,
    s2_train,
    s3_train,
    top_k=TOP_K
)

val_ground_truth = {
    k: v
    for k, v in ground_truth.items()
    if k in val_ids
}

stats = compute_blocking_recall(
    candidates,
    val_ground_truth
)

print_recall_report(stats)
```

The target is:

```text
Blocking Recall >= 95%
```

This is important because a true match removed during blocking cannot be recovered by the later matching model.

---

## 6. Large Dataset Handling

The datasets can contain millions of records, so running full TF-IDF nearest-neighbor search can require significant RAM and processing time.

During development, a small batch is therefore used first:

```text
500 records
    ↓
Test pipeline
    ↓
Check RAM and runtime
    ↓
Optimize blocking
    ↓
Increase dataset size
```

The 500-record run is only for **development and debugging**. It is not the final blocking-recall evaluation.

For final evaluation, the complete validation population should be processed.

---

## 7. Candidate Matching

After blocking reaches the required recall, the candidate pairs are passed to the matching stage.

For each candidate pair, features can be generated from:

* Business-name similarity
* Address similarity
* Token similarity
* Phonetic agreement
* Country agreement
* TF-IDF similarity
* Other normalized fields

The matching model then predicts whether the candidate pair represents the same business.

```text
Candidate Pair
      ↓
Feature Generation
      ↓
Matching Model
      ↓
Match / Non-Match
```

---

## 8. Output Files

The project produces two main outputs.

### Candidate pairs

```text
output/candidate_pairs.tsv
```

Contains the final candidate pairs passed to the matching stage.

### Final matches

```text
output/matching_results.tsv
```

Contains the final predicted S2/S3 matches for each S1 entity.

Every S1 entity should have an output row, including entities with no matches.

---

## 9. Development Strategy

The project is developed progressively:

```text
500 records
     ↓
Check correctness
     ↓
Check memory/runtime
     ↓
Optimize blocking
     ↓
5,000 records
     ↓
Larger validation set
     ↓
Full validation
     ↓
Blocking recall >= 95%
     ↓
Train final matching model
     ↓
Generate final predictions
```

This prevents expensive full-dataset processing while the blocking implementation is still being developed.

---

## 10. Project Structure

A recommended structure is:

```text
project/
│
├── data/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
│
├── src/
│   ├── normalization.py
│   ├── blocking.py
│   ├── matching.py
│   └── evaluation.py
│
├── output/
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
│
├── requirements.txt
└── README.md
```

---

## 11. Main Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Jellyfish
* TF-IDF
* Nearest Neighbors

---

## 12. Final Objective

The final system should:

1. Normalize noisy business records.
2. Partition records by country.
3. Generate high-recall candidate pairs.
4. Achieve at least **95% blocking recall** on validation data.
5. Score candidate pairs using a matching model.
6. Generate final entity-resolution predictions.
7. Produce valid `candidate_pairs.tsv` and `matching_results.tsv` files.

The key design goal is to achieve high blocking recall while keeping memory usage and runtime manageable on the large datasets.
