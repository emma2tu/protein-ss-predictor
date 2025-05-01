# 🧬 CM121 Project Pipeline — Pseudocode Overview

## `main.py` — Entry Point

Controls high-level workflow. Does **not** import `ProteinPredictionPipeline.py` directly.

### Functions:
- `process_all_files()`  
  → Loads raw data and creates filtered training, validation, and test sets  
- `train_pipeline()`  
  → Embeds training data and trains classifier  
- `validate_pipeline()`  
  → Runs classifier on validation set and evaluates performance  
- `make_predictions()`  
  → Generates predictions on test data for submission

---

## 1️⃣ Protein Data Loading & Structuring

### `ProteinDataProcessor.py` — Input Parsing and Filtering

#### 🔹 Raw File Processing

- `process_train_tsv(train_tsv)`
  - **Input**: `train.tsv`
  - **Output**:  
    - `train_df`: Pandas DataFrame of residue labels  
    - `train_tsv_dict`: `{ protein_name → list of secondary structure labels }`  
  - Pads label list so that residue index matches list index (e.g., residue 6 → index 5)

- `process_test_tsv(test_tsv)`
  - **Input**: `test.tsv`
  - **Output**:  
    - `test_df`: Pandas DataFrame of residue IDs  
    - `test_tsv_dict`: `{ protein_name → list of residue IDs }`

- `process_sequences_fasta(sequences_fasta)`
  - **Input**: `sequences.fasta`
  - **Output**:  
    - `raw_sequences_dict`: `{ protein_name → amino acid sequence (string) }`

#### 🔹 Filtering and Splitting

- `filter_training_sequences()`
  - **Filters**:  
    - Proteins not in `train_tsv_dict`  
    - Proteins with sequence length < number of labeled residues  
  - **Output**:  
    - `training_sequences_dict`: `{ protein_name → sequence }`

- `filter_testing_sequences()`
  - **Filters**: Proteins not in `test_tsv_dict`  
  - **Output**:  
    - `testing_sequences_dict`: `{ protein_name → sequence }`

- `make_training_validation_sets(val_every=5)`
  - **Splits**: `training_sequences_dict` into 80/20 training/validation sets  
    - Every 5th valid sequence goes to validation  
  - **Filters**: out sequences with invalid residues (not in `config.residue_13_dict`)  
  - **Output**:  
    - `training_set`: `{ protein_name → (sequence, label list) }`  
    - `validation_set`: `{ protein_name → (sequence, label list) }`

- `make_testing_set()`
  - Filters invalid test sequences and stores them in:
  - **Output**:  
    - `testing_set`: `{ protein_name → sequence }`

#### 📦 Core Data Structures:

| Name                      | Type                          | Description |
|---------------------------|-------------------------------|-------------|
| `train_tsv_dict`          | dict[str, list[str]]          | Protein → secondary structure list (with padding) |
| `test_tsv_dict`           | dict[str, list[str]]          | Protein → residue ID list |
| `raw_sequences_dict`      | dict[str, str]                | Protein → full amino acid sequence |
| `training_sequences_dict` | dict[str, str]                | Filtered training sequences |
| `testing_sequences_dict`  | dict[str, str]                | Filtered testing sequences |
| `training_set`            | dict[str, tuple[str, list]]   | Training protein → (sequence, labels) |
| `validation_set`          | dict[str, tuple[str, list]]   | Validation protein → (sequence, labels) |
| `testing_set`             | dict[str, str]                | Final test sequences |

> ⚠️ **Note**: If a sequence is shorter than the max residue index in `train.tsv` or `test.tsv`, skip it — ESM output won't match and will cause index errors.  
> During postprocessing, use `'.'` as a placeholder for positions that don’t have predictions.

---

## 2️⃣ Feature Extraction & Classification

### `ProteinPredictionPipeline.py` — ESM + Classifier

#### (1) Preprocessing & Embedding

- `tokenize_and_embed(sequence)`
  - **Input**: Amino acid sequence (string)
  - **Output**: ESM embedding tensor (logits) for each residue

#### (3) Classification

- `predict_secondary_structure(sequence)`
  - **Input**: Sequence (string)
  - **Output**: List of predicted structure labels (one per residue)

#### (4) Postprocessing

- `align_predictions_to_ids(sequence, predictions, residue_ids)`
  - **Purpose**: Aligns model predictions with expected residue IDs from `test.tsv`
  - **Handles**:
    - Out-of-bounds indices
    - Fills gaps using `"."`

---

### `trainer.py` — Training the Random Forest Classifier

#### (2) Generate Training Embeddings

- `generate_logits(sequences_dict)`
  - **Input**: `{ protein → sequence }`
  - **Output**: `{ protein → [embedding vectors per residue] }`
  - Applies ESM model to convert amino acid sequences into per-residue embeddings

#### (2) Train Classifier

- `train_classifier(logits_dict, labels_dict)`
  - **Input**:  
    - `logits_dict`: `{ protein → embedding vectors }`  
    - `labels_dict`: `{ protein → list of secondary structure labels }`  
  - **Output**:  
    - Trained classifier (e.g., `RandomForestClassifier`)
  - Combines all residue embeddings across proteins into a matrix `X` and labels into `y`

---

## 3️⃣ Validation

### `validator.py` — Model Evaluation

- `validate()`
  - Runs prediction pipeline on the validation set  
  - Compares output to true labels  
  - Returns accuracy or confusion matrix

---

## 4️⃣ Submission Pipeline

### `feeder.py` — Final Prediction

- `run_test_pipeline()`
  - Runs prediction pipeline on `testing_set`  
  - Generates `prediction.csv` with:
    - `id`, `prediction`
  - Zips for Codabench submission
