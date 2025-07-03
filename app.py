import os, requests, hashlib, streamlit as st, logging

logging.basicConfig(level=logging.INFO)

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # optional / for gated models
CHUNK = 2**20                                     # 1 MB

def _download(url, dest):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if chunk:
                    f.write(chunk)

def _needs_download(path, expected_mb: int):
    return (not os.path.exists(path)) or os.path.getsize(path) < expected_mb * 2**20 * 0.9

def download_models():
    os.makedirs("models", exist_ok=True)

    models = [
        ("https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-GGUF/resolve/main/"
         "tinyllama-1.1B-chat.Q4_K_M.gguf",
         "models/tinyllama-1.1B-chat.Q4_K_M.gguf",
         520),                       # MB
        ("https://huggingface.co/robinjia/emo_mirror_assets/raw/main/voice_svm.joblib",
         "models/voice_svm.joblib",
         3),                         # MB
    ]

    with st.spinner("⏬ Downloading models (first run only)…"):
        for url, path, size_mb in models:
            if _needs_download(path, size_mb):
                logging.info(f"Downloading {os.path.basename(path)}…")
                _download(url, path)
                logging.info(f"Saved → {path}")
            else:
                logging.info(f"{os.path.basename(path)} already present.")

# Call once when script starts
download_models()




# app.py
import streamlit as st, datetime as dt
from pipelines import text_distilbert as txt
from pipelines import voice_osmile as voc
from pipelines import face_fer as fac
from pipelines import fuse
from brain import llama_cpp_reply as bot
from brain import memory
from components.audio_rec import audio_recorder
from components.mood_chart import draw_chart

st.set_page_config("🧬 Emotion Mirror (CPU‑only)", layout="wide")
mem = memory.ChatMemory()                 # session‑level memory

st.title("🧬 Emotion Mirror – Reflect & Chat (CPU Edition)")

# ────────────────────────────────────────────────────────────
# Layout: chat left, media right
col_chat, col_media = st.columns([3, 2])

# ---- TEXT INPUT -----------------------------------------------------------
with col_chat:
    user_text = st.chat_input("Tell me what's on your mind…")
    if user_text:
        label_t, probs_t = txt.detect(user_text)
        mem.add("user", user_text)
        st.chat_message("user").write(user_text)
        modal_logits = {"text": probs_t}

# ---- VOICE INPUT ----------------------------------------------------------
with col_media:
    wav_bytes = audio_recorder("🎙️  Hold to record voice", pause_threshold=1.0)
    if wav_bytes:
        vlabel, vprobs = voc.detect(wav_bytes)
        st.success(f"Voice emotion → {vlabel}")
        modal_logits["voice"] = vprobs

# ---- WEBCAM INPUT ---------------------------------------------------------
with col_media:
    frame = st.camera_input("📸  Snap webcam photo")
    if frame is not None and st.button("Analyse face"):
        flabel, fprobs = fac.detect(frame.getvalue())
        st.success(f"Face emotion → {flabel}")
        modal_logits["face"] = fprobs

# ---- GENERATE AI REPLY ----------------------------------------------------
if mem.last_is_user():
    idx, fused = fuse.fuse(modal_logits)
    emo_tag = fuse.LABELS[idx]            # human‑readable label
    reply = bot.reply(mem.history, emo_tag)
    mem.add("ai", reply)
    st.chat_message("assistant").markdown(reply)

# ---- (Optional) Mood trend -----------------------------------------------
with st.expander("📊  Mood trend"):
    draw_chart(mem.moodlog)
