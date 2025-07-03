# pipelines/fuse.py
import numpy as np
LABELS = ["negative", "positive"]   # adapt if using multi‑label model

def fuse(modal_logits: dict[str, np.ndarray]):
    # Simple late fusion: weight by max confidence
    weights = {m: lg.max() for m, lg in modal_logits.items()}
    total   = sum(weights.values()) or 1
    fused   = sum(weights[m] * modal_logits[m] for m in modal_logits) / total
    idx     = int(np.argmax(fused))
    return idx, fused
