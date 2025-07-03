# components/mood_chart.py
import pandas as pd, streamlit as st

def draw_chart(log):
    """log list[(ts,val,ar)] – placeholder chart"""
    if not log:
        st.info("Mood log empty – start chatting!")
        return
    df = pd.DataFrame(log, columns=["ts","val","ar"]).set_index("ts")
    st.line_chart(df)
