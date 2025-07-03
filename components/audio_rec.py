# components/audio_rec.py
import streamlit as st
from typing import Optional

def audio_recorder(label="Record", pause_threshold: float = 1.0) -> Optional[bytes]:
    """
    Returns WAV bytes or None. Uses st.file_uploader fallback if audio_recorder
    component isn't available.
    """
    try:
        from streamlit_audio_recorder import st_audiorec
        wav = st_audiorec(label=label, pause_threshold=pause_threshold)
        return wav
    except ModuleNotFoundError:
        st.info("Voice recorder not installed; upload a WAV file instead.")
        file = st.file_uploader("Upload WAV", type=["wav"])
        if file:
            return file.read()
    return None
