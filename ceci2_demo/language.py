"""Constrained instruction-model forecast conditional on a Fed statement."""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PROMPT = (
    "Given this central bank statement, classify the broad US equity market's direction "
    "over the next five trading sessions. A = DOWN, B = FLAT, C = UP. "
    "Answer with one letter.\n\n"
)


def classify(folder, documents, family="qwen"):
    local = folder / family
    identifier, revision = (
        ("Qwen/Qwen2.5-0.5B-Instruct", REVISION)
        if family == "qwen"
        else (
            "ibm-granite/granite-3.1-1b-a400m-instruct",
            "0da7a48b0276d500ce5922fd2b33944091fc6c09",
        )
    )
    source = str(local) if (local / "model.safetensors").exists() else identifier
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(source, revision=revision).eval()
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model.to(device)
    ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in "ABC"]
    if any(len(tokens) != 1 for tokens in ids):
        raise ValueError("Each response option must be a single token")
    outcome = []
    with torch.inference_mode():
        for document in documents:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": PROMPT + document}],
                tokenize=False,
                add_generation_prompt=True,
            )
            tokens = tokenizer(prompt, return_tensors="pt").to(device)
            logits = (
                model(**tokens, use_cache=False)
                .logits[0, -1, [i[0] for i in ids]]
                .float()
            )
            outcome.append(logits.softmax(-1).cpu().numpy())
    return np.stack(outcome)
