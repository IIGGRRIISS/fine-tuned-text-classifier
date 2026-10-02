# Fine-Tuned Text Classifier: DistilBERT Sentiment

A supervised NLP classifier that fine-tunes a pretrained DistilBERT transformer for binary sentiment classification (positive / negative) on the IMDb movie review dataset, using Hugging Face Transformers and PyTorch.

**🤗 Model on Hugging Face Hub:** [IIGGRRIISS/distilbert-imdb-sentiment](https://huggingface.co/IIGGRRIISS/distilbert-imdb-sentiment)

---

## Results

Evaluated on a held-out test set of 500 IMDb reviews:

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.872 |
| Precision | 0.864 |
| Recall    | 0.878 |
| F1-score  | 0.871 |

Per-class performance:

| Class    | Precision | Recall | F1   | Support |
|----------|-----------|--------|------|---------|
| negative | 0.88      | 0.87   | 0.87 | 254     |
| positive | 0.86      | 0.88   | 0.87 | 246     |

> Trained on a 2,000-sample subset of IMDb for 2 epochs on a free Colab T4 GPU. Training on the full dataset typically reaches ~92–93% accuracy.

---

## Pipeline

```text
Raw text
  → Tokenizer (WordPiece)
  → input_ids + attention_mask (max_len = 256)
  → DistilBERT encoder (pretrained, 66M params)
  → Classification head (2 labels)
  → Softmax → Prediction + confidence
```

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers (`distilbert-base-uncased`)
- Hugging Face Datasets (`stanfordnlp/imdb`)
- Hugging Face Trainer
- scikit-learn (metrics)

---

## Project Structure

```text
.
├── train.py            # Fine-tune DistilBERT on IMDb
├── evaluate.py         # Compute accuracy, precision, recall, F1
├── predict.py          # Run inference on custom text
└── requirements.txt    # Dependencies
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train from scratch

```bash
python train.py
```

### 3. Evaluate

```bash
python evaluate.py
```

### 4. Run inference

```bash
python predict.py
```

---

## Load the Model Directly from Hugging Face

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_id = "IIGGRRIISS/distilbert-imdb-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.eval()

text = "I absolutely loved this product!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits, dim=-1)[0]

label = ["NEGATIVE", "POSITIVE"][int(torch.argmax(probs))]
print(f"{label} ({probs.max().item() * 100:.2f}%)")
# → POSITIVE (91.47%)
```

---

## Example Predictions

| Input                                          | Prediction | Confidence |
|------------------------------------------------|------------|------------|
| "I absolutely loved this product!"             | POSITIVE   | 91.5%      |
| "The service was terrible and extremely slow." | NEGATIVE   | 90.5%      |
| "It was okay, nothing special."                | NEGATIVE   | 81.6%      |
| "Best movie I've seen in years."               | POSITIVE   | 87.3%      |
| "Waste of time and money."                     | NEGATIVE   | 93.3%      |

---

## Concepts Demonstrated

- Transfer learning (reusing pretrained language representations)
- Fine-tuning a transformer on a downstream task
- Tokenization and attention masking
- Supervised learning loop: forward pass → loss → backprop → optimizer step
- Model evaluation with per-class metrics
- Saving and loading a trained model as a portable artifact
- Hosting a model on the Hugging Face Hub

---

## Author

**Syed Ibrahim Ali**
GitHub: [IIGGRRIISS](https://github.com/IIGGRRIISS)
