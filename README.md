# Document Classification using Transformers (BERT)

## Overview
# Document Classification using Transformer Models (BERT)

This project classifies long legal documents into predefined categories using Transformer models (BERT) with chunking and attention pooling.
---

## Highlights
- **Dataset**: LexGLUE ECtHR (9,000 training, 1,000 validation, 1,000 test)
- **Classes**: 10 categories
- **Techniques**: Chunking with overlap, Attention Pooling, Class-weighted Cross-Entropy
- **Results**: Accuracy: 70.41%, Weighted F1: 70.60%


---

## Workflow
1. Data Preprocessing (tokenization, chunking, filtering)
2. Model: BERT base with custom Attention Pooling
3. Training: AdamW, Gradient Accumulation, Learning Rate Warm-up
4. Evaluation: Accuracy, Weighted F1, Confusion Matrix
5. K-Fold Cross Validation for robust performance


---

## How to Run
1. Clone the repository
   git clone https://github.com/yourusername/document-classification-transformers.git
2. Install dependencies
   pip install -r requirements.txt
3. Open and run `document_classifier.ipynb`
   
---

## Results
- Test Accuracy: 70.41%
- Weighted F1: 70.60%
- Confusion matrix shows strong performance on frequent classes, weaker on rare ones.

## Challenges
- Long document handling (solved with chunking & attention pooling)
- Class imbalance (addressed with class weights)
- GPU memory limits (handled using gradient accumulation)

---

## Detailed Report
For a full explanation of preprocessing, model architecture, training pipeline, and cross-validation, see:
[Full Project Report](PROJECT_REPORT.md)



