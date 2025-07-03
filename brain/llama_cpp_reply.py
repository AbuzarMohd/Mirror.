# brain/llama_cpp_reply.py
from llama_cpp import Llama
import os, functools

MODEL_PATH = os.path.join("models", "tinyllama-1.1B-chat.Q4_K_M.gguf")

@functools.lru_cache(maxsize=1)
def _load():
    # adjust n_threads to your CPU cores (MacBook i5 ≈ 4 threads)
    return Llama(model_path=MODEL_PATH, n_ctx=1536, n_threads=4)

def reply(history: list[tuple[str,str]], emotion: str) -> str:
    convo = "\n".join(f"{r.upper()}: {m}" for r, m in history[-6:])
    prompt = (f"[INST]You are 'Emotion Mirror', an empathetic AI companion.\n"
              f"Detected emotion: {emotion}\n{convo}\nAI:[/INST]")
    out = _load()(prompt, max_tokens=140, stop=["[/INST]"])
    return out["choices"][0]["text"].strip()
