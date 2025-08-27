# Document Classification using Transformer Models (BERT)

## Introduction
Classifying long legal documents into predefined categories is a challenging task due to the length and complexity of legal texts. Transformer-based models, such as **BERT** and **DistilBERT**, have emerged as state-of-the-art approaches for text classification because they can encode semantic context across long sequences.

### Goal
Classify long legal documents into predefined categories using Transformer models, specifically **BERT** or **DistilBERT**.

### Project Objectives
- Tokenize and preprocess long legal documents
- Apply chunking with overlap to handle sequences longer than the Transformer model limit (512 tokens)
- Use attention pooling to aggregate chunk-level embeddings into document-level embeddings
- Fine-tune BERT for multi-class classification
- Implement gradient accumulation and learning rate warm-up for efficient training
- Evaluate using accuracy, weighted F1-score, and classification reports

---

## 1. Setup and Data Loading
- Libraries: `transformers`, `datasets`, `accelerate`, `scikit-learn`
- Dataset: **LexGLUE ECtHR**
  - Train: 9,000 documents
  - Validation: 1,000 documents
  - Test: 1,000 documents
  - Categories: 10

Initial analysis revealed a significant class imbalance, with some categories having many more examples than others.

---

## 2. Data Preprocessing

### Tokenization
The documents, which are lists of text snippets, were joined into single strings and then tokenized using the `bert-base-uncased` tokenizer.

### Chunking with Overlap
- Chunk size: 512 tokens
- Overlap: 50 tokens
- Each chunk was assigned a `doc_id` to keep track of its original document.

### Filtering
Chunks with empty labels were removed from the dataset.

---

## 3. Dataset and DataLoader Preparation
- Implemented a custom PyTorch `Dataset` class: **ChunkedLegalDataset**
  - Prepares `input_ids`, `attention_mask`, and `labels` (taking the first label if multiple were present)
- Created `DataLoader` objects for training, validation, and test sets
- Used a custom `collate_fn` to handle sequence padding within each batch

---

## 4. Model Architecture
- **Base Model:** `bert-base-uncased`
- **Pooling Layer:** Custom Attention Pooling
- **Classifier:** Linear layer with Dropout

Workflow:


---

## 5. Training
- Optimizer: **AdamW** with `lr = 1e-5`
- Loss: **Cross-Entropy with class weights** (to address class imbalance)
- Epochs: **5**
- Gradient accumulation: **4 steps** (effective batch size = 32)
- Mixed precision training with `torch.cuda.amp.GradScaler` and `autocast`
- Learning rate warm-up: First 10% of steps
- Model saved after each epoch
- Metrics: Accuracy and Weighted F1-score

---

## 6. Evaluation
- Test Accuracy: **0.7041**
- Weighted F1-score: **0.7060**
- Generated:
  - Classification report
  - Confusion matrix (to visualize misclassifications)

---

## 7. Class Distribution Analysis
- Class imbalance observed:
  - Most frequent: **Class 3**
  - Least frequent: **Class 5**
- Addressed using **class weights** during training

---

## 8. K-Fold Cross-Validation
Implemented to:
- Improve robustness
- Reduce bias and variance
- Select the best performing fold based on validation F1-score

### Procedure:
- K = 5 folds
- Applied same preprocessing, chunking, and attention pooling for each fold
- Monitored validation performance after each epoch
- Saved best model per fold
- Final evaluation on the held-out test set ensured unbiased results

---

## 9. Inference
- User-defined text is tokenized and split into chunks
- Attention pooling aggregates embeddings
- Classifier predicts the final category

---

## 10. Challenges and Limitations
- Long documents required chunking → increased computation and memory requirements
- Class imbalance affected rare categories → mitigated using class weights
- GPU memory constraints → solved using gradient accumulation
- Slight overfitting observed due to relatively small dataset
- Attention pooling added computational cost but improved document-level accuracy

---

