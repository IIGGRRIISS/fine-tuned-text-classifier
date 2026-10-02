"""
Fine-tune DistilBERT for binary sentiment classification on IMDb.
"""

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./fine_tuned_distilbert_imdb"
TRAIN_SIZE = 2000
TEST_SIZE = 500
MAX_LEN = 256


def load_and_shrink_dataset():
    dataset = load_dataset("stanfordnlp/imdb")
    train = dataset["train"].shuffle(seed=42).select(range(TRAIN_SIZE))
    test = dataset["test"].shuffle(seed=42).select(range(TEST_SIZE))
    return train, test


def build_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_dataset(dataset, tokenizer):
    def _tok(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )

    return dataset.map(_tok, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    train_raw, test_raw = load_and_shrink_dataset()
    tokenizer = build_tokenizer()

    train_ds = tokenize_dataset(train_raw, tokenizer)
    test_ds = tokenize_dataset(test_raw, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("\nFinal evaluation:", trainer.evaluate())

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved model + tokenizer to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()