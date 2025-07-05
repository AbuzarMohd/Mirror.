'''
import os
import logging
import requests
import streamlit as st
from transformers import pipeline

# ─────────────────────────────  0 · Setup  ─────────────────────────────

st.set_page_config(page_title="Emotion Mirror Chat", layout="wide")
logging.basicConfig(level=logging.INFO)

CHUNK = 2**20  # 1MB


def _need(path, mb):
    return not os.path.exists(path) or os.path.getsize(path) < mb * 0.9 * CHUNK


def _download(url: str, dest: str):
    """Stream download without auth (public URL)."""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as fp:
            for chunk in r.iter_content(CHUNK):
                if chunk:
                    fp.write(chunk)


def ensure_voice_svm():
    os.makedirs("models", exist_ok=True)
    url  = "https://huggingface.co/datasets/AbuzarMohd/emo_mirror_assets/resolve/main/voice_svm.joblib"
    path = "models/voice_svm.joblib"
    if _need(path, 3):
        with st.spinner("⏬ Downloading voice‑SVM (first run)…"):
            logging.info("Downloading voice_svm.joblib …")
            _download(url, path)
            logging.info("Saved → models/voice_svm.joblib")


ensure_voice_svm()

# ─────────────────────────────  2 · TinyLLaMA via Transformers  ─────

@st.cache_resource(show_spinner="🔄 Loading TinyLLaMA…")
def load_pipe():
    return pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype="auto",
        device_map="auto"
    )

pipe = load_pipe()

# ─────────────────────────────  3 · UI Layout  ─────────────────────────────

st.title("🪞 Emotion Mirror (TinyLLaMA Powered)")

system_prompt = st.text_input("System Prompt", value="You are a kind emotional support chatbot.")
user_input = st.text_area("Your Message", height=200)

if st.button("Generate Response"):
    if user_input.strip():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        prompt = pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        with st.spinner("Thinking…"):
            output = pipe(
                prompt,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )
        response = output[0]["generated_text"].split("</s>")[-1].strip()
        st.markdown("### 🤖 Response")
        st.success(response)
'''
'''
import os, logging, streamlit as st
from transformers import pipeline
from pipelines import voice_osmile  as voc   # ← NEW SpeechBrain pipeline
# If you still want text / face analysis, import them here
 from pipelines import text_distilbert as txt
 from pipelines import face_fer         as fac

# ─────────────────────────────  Setup  ─────────────────────────────
st.set_page_config(page_title="Emotion Mirror Chat", layout="wide")
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────  TinyLLaMA  ─────────────────────────
@st.cache_resource(show_spinner="🔄 Loading TinyLLaMA…")
def load_pipe():
    return pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype="auto",
        device_map="auto"          # CPU on Streamlit Cloud
    )

llm = load_pipe()

# ─────────────────────────────  UI  ─────────────────────────────
st.title("🪞 Emotion Mirror – TinyLLaMA & SpeechBrain")

system_prompt = st.text_input(
    "System Prompt",
    value="You are a kind emotional‑support chatbot.",
)

# ========== Voice recorder ==========
wav_bytes = st.file_uploader("🎙️ Upload or record a WAV file for emotion check", type=["wav"])
if wav_bytes:
    vlabel, vscores = voc.detect(wav_bytes.read())
    st.success(f"Detected voice emotion: **{vlabel}**")

# ========== Text chat ==========
user_input = st.text_area("Your Message", height=200)
if st.button("Generate Response") and user_input.strip():
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    prompt = llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    with st.spinner("Thinking…"):
        output = llm(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )
    response = output[0]["generated_text"].split("</s>")[-1].strip()
    st.markdown("### 🤖 Response")
    st.success(response)
'''
# app.py
import os, logging, tempfile, numpy as np
import streamlit as st
from transformers import pipeline

# modality pipelines
from pipelines import text_distilbert as txt
from pipelines import face_fer         as fac
from pipelines import voice_osmile         as voc  # SpeechBrain wav2vec2 model

from components.audio_rec  import audio_recorder
from components.mood_chart import draw_chart

# ────────────────────────────  CONFIG  ────────────────────────────
st.set_page_config(page_title="🪞 Emotion Mirror – All Modalities", layout="wide")
logging.basicConfig(level=logging.INFO)

# ─────────────────────  TinyLLaMA chat model  ─────────────────────
@st.cache_resource(show_spinner="🔄 Loading TinyLLaMA …")
def load_chat():
    return pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype="auto",
        device_map="auto",   # CPU on Streamlit Cloud
    )

chat_pipe = load_chat()

# ────────────────────────────  UI  ────────────────────────────────
st.title("🪞 Emotion Mirror – Text · Voice · Face")

modal_logits: dict[str, np.ndarray] = {}
user_text   = ""

col_text, col_voice, col_face = st.columns(3)

# TEXT INPUT  ───────────────────────────────────────────────────────
with col_text:
    user_text = st.text_area("📝 Type your message")
    if st.button("Analyse Text") and user_text.strip():
        t_label, t_probs = txt.detect(user_text)
        st.success(f"Text emotion → {t_label}")
        modal_logits["text"] = t_probs

# VOICE INPUT  ──────────────────────────────────────────────────────
with col_voice:
    wav_bytes = audio_recorder("🎙️ Record / Upload Voice")
    if wav_bytes:
        v_label, v_probs = voc.detect(wav_bytes)
        st.success(f"Voice emotion → {v_label}")
        modal_logits["voice"] = v_probs

# FACE INPUT  ───────────────────────────────────────────────────────
with col_face:
    frame = st.camera_input("📸 Take a photo")
    if frame is not None and st.button("Analyse Face"):
        f_label, f_probs = fac.detect(frame.getvalue())
        st.success(f"Face emotion → {f_label}")
        modal_logits["face"] = f_probs

# ───────────────  FUSE EMOTIONS & GENERATE REPLY  ─────────────────
if modal_logits:
    # average probabilities across available modalities
    avg_probs = np.mean(list(modal_logits.values()), axis=0)
    idx = int(np.argmax(avg_probs))
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    detected_emotion = emotions[idx]

    messages = [
        {"role": "system", "content": f"You are a compassionate AI. Detected user emotion: {detected_emotion}."},
        {"role": "user",   "content": user_text or "(User provided voice or face input)"},
    ]
    prompt = chat_pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    with st.spinner("🤔 Reflecting …"):
        out = chat_pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.9)
    reply = out[0]["generated_text"].split("</s>")[-1].strip()

    st.markdown("### 🤖 Response")
    st.success(reply)

# ─────────────────────────  MOOD CHART  ────────────────────────────
with st.expander("📊 Mood Trends"):
    draw_chart([])
