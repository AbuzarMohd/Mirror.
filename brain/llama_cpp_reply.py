import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Emotion Mirror Chat", layout="wide")

@st.cache_resource
def load_model():
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype="auto",
        device_map="auto"
    )
    return pipe

pipe = load_model()

st.title("🪞 Emotion Mirror (TinyLLaMA Powered)")

system_prompt = st.text_input("System Prompt", value="You are a kind emotional support chatbot.")
user_input = st.text_area("Your Message", height=200)

if st.button("Generate Response"):
    if user_input.strip():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with st.spinner("Thinking..."):
            output = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        response = output[0]["generated_text"].split("</s>")[-1].strip()
        st.markdown("### 🤖 Response")
        st.success(response)
