# pipelines/face_fer.py
from fer import FER
import cv2, numpy as np, functools

@functools.lru_cache(maxsize=1)
def _det():
    return FER(mtcnn=True)

def detect(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    res = _det().detect_emotions(img)
    if not res:
        return "neutral", np.zeros(7)
    emo = res[0]["emotions"]
    label = max(emo, key=emo.get)
    logits = np.array(list(emo.values()))
    return label, logits
