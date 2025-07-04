# ─────────────────────────────  1 · VOICE‑SVM downloader  ─────────────────────────────
import os, requests, logging, streamlit as st
logging.basicConfig(level=logging.INFO)

CHUNK = 2**20  # 1 MB

def _download(url: str, dest: str):
    """Stream download without auth (public URL)."""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as fp:
            for chunk in r.iter_content(CHUNK):
                if chunk:
                    fp.write(chunk)

def _need(path: str, mb: int) -> bool:
    return (not os.path.exists(path)) or os.path.getsize(path) < mb * 0.9 * 2**20

def ensure_voice_svm():
    os.makedirs("models", exist_ok=True)
    url  = "https://raw.githubusercontent.com/robinjia/tmp-assets/main/voice_svm.joblib"
    path = "models/voice_svm.joblib"
    if _need(path, 3):
        with st.spinner("⏬ Downloading voice‑SVM (first run)…"):
            logging.info("Downloading voice_svm.joblib …")
            _download(url, path)
            logging.info("Saved → models/voice_svm.joblib")

ensure_voice_svm()

# ─────────────────────────────  2 · TinyLLaMA via Transformers  ───────────────────────
from transformers import pipeline

@st.cache_resource(show_spinner="🔌 Loading TinyLLaMA …")
def load_llm():
    return pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype="auto",     # bfloat16 if CPU supports, else fp32
        device_map="auto"       # CPU on Streamlit Cloud
    )

llm = load_llm()

# ─────────────────────────────  3 · Emotion pipelines & UI  ───────────────────────────
from pipelines import text_distilbert as txt
from pipelines import voice_osmile   as voc
from pipelines import face_fer       as fac
from pipelines import fuse
from brain     import memory
from components.audio_rec  import audio_recorder
from components.mood_chart import draw_chart

st.set_page_config("🪞 Emotion Mirror – TinyLLaMA", layout="wide")

modal_logits: dict[str, list] = {}
mem = memory.ChatMemory()

st.title("🪞 Emotion Mirror — TinyLLaMA Edition")

col_chat, col_media = st.columns([3, 2])

# ---- TEXT ------------------------------------------------------------------
with col_chat:
    user_text = st.chat_input("Share what's on your mind …")
    if user_text:
        lbl, prob = txt.detect(user_text)
        mem.add("user", user_text)
        st.chat_message("user").write(user_text)
        modal_logits["text"] = prob

# ---- VOICE -----------------------------------------------------------------
with col_media:
    wav = audio_recorder("🎙️ Record voice")
    if wav:
        vlab, vprob = voc.detect(wav)
        st.success(f"Voice emotion → {vlab}")
        modal_logits["voice"] = vprob

# ---- FACE ------------------------------------------------------------------
with col_media:
    snap = st.camera_input("📸 Webcam snapshot")
    if snap is not None and st.button("Analyse face"):
        flab, fprob = fac.detect(snap.getvalue())
        st.success(f"Face emotion → {flab}")
        modal_logits["face"] = fprob

# ---- LLM REPLY -------------------------------------------------------------
if mem.last_is_user() and modal_logits:
    idx, fused = fuse.fuse(modal_logits)
    emo = fuse.LABELS[idx]

    messages = [
        {"role": "system", "content": "You are Emotion Mirror, an empathetic AI."},
        {"role": "assistant", "content": f"I sense you feel {emo}."},
        {"role": "user", "content": user_text},
    ]
    prompt = llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    with st.spinner("🤔 Reflecting …"):
        out = llm(
            prompt,
            max_new_tokens=140,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
        )
    reply = out[0]["generated_text"].split("</s>")[-1].strip()
    mem.add("ai", reply)
    st.chat_message("assistant").markdown(reply)

# ---- Mood chart ------------------------------------------------------------
with st.expander("📊 Mood trend"):
    draw_chart(mem.moodlog)
