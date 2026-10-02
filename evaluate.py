"""
Evaluate the fine-tuned model on a held-out test set.
Reports accuracy, precision, recall, F1 per-class and overall.
"""

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "IIGGRRIISS/distilbert-imdb-sentiment"
MAX_LEN = 256
TEST_SIZE = 500


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()

    test = (
        load_dataset("stanfordnlp/imdb")["test"]
        .shuffle(seed=42)
        .select(range(TEST_SIZE))
    )

    all_preds, all_labels = [], []

    for example in test:
        inputs = tokenizer(
            example["text"],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = int(torch.argmax(logits, dim=-1))
        all_preds.append(pred)
        all_labels.append(example["label"])

    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=["negative", "positive"],
            digits=4,
        )
    )


if __name__ == "__main__":
    main()