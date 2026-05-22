import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=800,
    temperature=0.7,
    do_sample=True,
    return_full_text=False
)

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Be accurate and concise. "
    "If unsure about something, say so rather than guessing."
)

def chat(message: str, history: list[list[str]]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    result = pipe(formatted)
    return result[0]["generated_text"].strip()

demo = gr.ChatInterface(
    fn=chat,
    title="BenchMind — Llama 3.1 8B",
    description="OSS model backend for BenchMind evaluator",
    examples=[
        "What is the capital of France?",
        "Write a Python function to reverse a string.",
        "Explain transformer attention in simple terms.",
    ],
)

if __name__ == "__main__":
    demo.launch()
