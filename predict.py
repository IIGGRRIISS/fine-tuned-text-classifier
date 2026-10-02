"""
Load the fine-tuned model and predict sentiment for new text.

By default, loads from the Hugging Face Hub. Pass a local path to
use a locally saved model instead.
"""

import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "IIGGRRIISS/distilbert-imdb-sentiment"
LABELS = {0: "NEGATIVE", 1: "POSITIVE"}
MAX_LEN = 256


def load_model(model_id: str = MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    return model, tokenizer


def predict(text: str, model, tokenizer):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
    ).to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs))

    return {
        "text": text,
        "prediction": LABELS[pred_id],
        "confidence": round(float(probs[pred_id]) * 100, 2),
        "scores": {
            "negative": round(float(probs[0]) * 100, 2),
            "positive": round(float(probs[1]) * 100, 2),
        },
    }


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else MODEL_ID
    print(f"Loading model: {model_id}")
    model, tokenizer = load_model(model_id)

    samples = [
        "I absolutely loved this product!",
        "The service was terrible and extremely slow.",
        "It was okay, nothing special.",
        "Best movie I've seen in years.",
        "Waste of time and money.",
    ]

    for s in samples:
        r = predict(s, model, tokenizer)
        print(f"\n→ {r['prediction']} ({r['confidence']}%)")
        print(f"  {s}")
        print(f"  negative: {r['scores']['negative']}%  positive: {r['scores']['positive']}%")


if __name__ == "__main__":
    main()