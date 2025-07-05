'''
# pipelines/voice_osmile.py
import opensmile, joblib, numpy as np, tempfile, warnings, os

# initialise openSMILE once
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
# load pre‑trained SVM – replace with your own fine‑tune if desired
_SVM_PATH = os.path.join("models", "voice_svm.joblib")
if not os.path.isfile(_SVM_PATH):
    warnings.warn("voice_svm.joblib not found – voice emotion disabled")
    svm = None
else:
    svm = joblib.load(_SVM_PATH)

def detect(wav_bytes: bytes):
    if svm is None:
        return "unknown", np.zeros(7)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
        fp.write(wav_bytes); path = fp.name
    feats = smile.process_file(path).iloc[0].values
    probs = svm.predict_proba([feats])[0]
    label = svm.classes_[int(np.argmax(probs))]
    return label, probs
'''

"""
Voice‑emotion detection using SpeechBrain wav2vec2 ‘SUPERB‑ER’ model.
No private SVM file, no OpenSMILE.
"""

from functools import lru_cache
from transformers import pipeline
import numpy as np
import tempfile
import soundfile as sf  # ensures wav is readable

@lru_cache(maxsize=1)
def _pipe():
    # CPU‑only pipeline; ~100 MB weights auto‑downloaded & cached
    return pipeline(
        "audio-classification",
        model="speechbrain/wav2vec2-base-superb-er",
        device="cpu",
    )

def detect(wav_bytes: bytes):
    # Write bytes to temp wav so pipeline can load
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
        fp.write(wav_bytes)
        wav_path = fp.name

    result = _pipe()(wav_path, top_k=5)
    # `result` is list[dict[label,score]] ordered by score desc
    label = result[0]["label"]
    scores = np.array([item["score"] for item in result])
    return label, scores
