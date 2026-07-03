import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, model_path="gpt2-finetuned", max_length=150, temperature=0.8, top_k=50, top_p=0.95):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()

    inputs = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    prompts = [
        "Once upon a time",
        "The scientist looked through the microscope and discovered",
        "In the future, artificial intelligence will",
    ]

    model_path = "gpt2-finetuned"

    if not os.path.exists(model_path):
        print(f"Fine-tuned model not found at '{model_path}'. Using base GPT-2.")
        model_path = "gpt2"

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 60)
        result = generate_text(prompt, model_path=model_path)
        print(result)
        print("=" * 60)
