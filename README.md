# PRODIGY_GA_1 — Text Generation with GPT-2

Fine-tune GPT-2 on a custom dataset to generate coherent, contextually relevant text.

## Setup

```bash
pip install -r requirements.txt
```

## Training

Fine-tune GPT-2 on the provided sample dataset:

```bash
python train.py
```

The fine-tuned model will be saved to `gpt2-finetuned/`.

## Generation

Generate text from prompts using the fine-tuned model:

```bash
python generate.py
```

## Dataset

Training data is in `data/train.txt` — a collection of paragraphs about AI and NLP. Replace with your own text to adapt the model's style and domain.

## Files

| File | Purpose |
|------|---------|
| `train.py` | Fine-tunes GPT-2 on a text corpus |
| `generate.py` | Generates text from prompts using the trained model |
| `data/train.txt` | Sample training data |
| `requirements.txt` | Python dependencies |
