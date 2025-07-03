# pipelines/text_distilbert.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, numpy as np, functools

_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"   # lightweight
@functools.lru_cache(maxsize=1)
def _load():
    tok = AutoTokenizer.from_pretrained(_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(_MODEL)
    mdl.eval()
    return tok, mdl

def detect(text: str):
    tok, mdl = _load()
    inputs = tok(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = mdl(**inputs).logits[0]
    probs = torch.softmax(logits, dim=0).cpu().numpy()
    label = "positive" if probs[1] > probs[0] else "negative"
    return label, probs           # return logits for fusion
