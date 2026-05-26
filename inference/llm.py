"""Sequential HuggingFace causal-LM wrapper.

Loads a model + tokenizer once (lazy singleton) and exposes a single
`generate_text(prompt, ...)` call used by the rest of the inference loop.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_LOCK = threading.Lock()
_CACHE = {}


def _load(model_name: str, device: str, dtype: torch.dtype):
    key = (model_name, device, str(dtype))
    if key in _CACHE:
        return _CACHE[key]
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
        )
        model.eval()
        _CACHE[key] = (tok, model)
        return tok, model


def generate_text(
    prompt: str,
    model_name: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    max_retries: int = 2,
) -> str:
    """Greedy (or sampled) generation. Returns text after the prompt."""
    tok, model = _load(model_name, device, dtype)

    chat = [{"role": "user", "content": prompt}]
    if hasattr(tok, "apply_chat_template"):
        try:
            inputs = tok.apply_chat_template(
                chat, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
        except Exception:
            inputs = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    else:
        inputs = tok(prompt, return_tensors="pt").input_ids.to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": max(temperature, 1e-5) if temperature > 0 else None,
        "pad_token_id": tok.pad_token_id,
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    err: Optional[Exception] = None
    for _ in range(max_retries + 1):
        try:
            with torch.no_grad():
                out = model.generate(inputs, **gen_kwargs)
            full = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
            return full.strip()
        except Exception as e:
            err = e
            time.sleep(1)
    raise RuntimeError(f"generate_text failed after retries: {err}")
