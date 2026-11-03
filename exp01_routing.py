"""Exp01: a per-language routing atlas of Qwen3-30B-A3B.

Question it answers first: how concentrated is expert use per language, layer by layer, and which experts are chosen
together? These counts feed the placement simulation in analyze.py.

Run (Modal, one H100, about 30-45 minutes the first time because the weights are downloaded):
    modal run exp01_routing.py
Writes results/routing_qwen3_30b_a3b.npz here and a copy to the Modal volume irb08-exp01.
Text: the first long articles of Wikipedia (20231101 dump) in ten languages, 96 sequences of 512 tokens each. The first
48 sequences of every language fit the placement, the last 48 evaluate it.
"""
import json
import pathlib
import time

import modal

MODEL = "Qwen/Qwen3-30B-A3B"
LANGS = {"en": "English", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
         "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi"}
SEQ_LEN = 512
SEQS_PER_LANG = 96
BATCH = 8

app = modal.App("irb08-exp01-routing")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.51.3", "accelerate==1.6.0", "datasets==3.5.0", "numpy==2.1.3",
                 "huggingface_hub[hf_transfer]==0.30.2", "safetensors==0.5.3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/vol/hf"})
)
vol = modal.Volume.from_name("irb08-exp01", create_if_missing=True)


@app.function(image=image, gpu="H100", volumes={"/vol": vol}, timeout=3 * 3600, memory=65536)
def run() -> bytes:
    import io

    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    path = snapshot_download(MODEL, local_dir="/vol/models/qwen3-30b-a3b",
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "merges.txt", "vocab.json"])
    vol.commit()
    print(f"weights ready after {time.time() - t0:.0f} s", flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    cfg = model.config
    n_layers, n_exp, top_k = cfg.num_hidden_layers, cfg.num_experts, cfg.num_experts_per_tok
    print(f"model loaded after {time.time() - t0:.0f} s: layers={n_layers} experts={n_exp} top_k={top_k}", flush=True)

    langs = list(LANGS)
    counts = np.zeros((2, len(langs), n_layers, n_exp), dtype=np.int64)  # [fit or eval, language, layer, expert]
    cooc = np.zeros((len(langs), n_layers, n_exp, n_exp), dtype=np.int32)  # experts chosen together, fit half only
    meta = {"model": MODEL, "langs": LANGS, "seq_len": SEQ_LEN, "seqs_per_lang": SEQS_PER_LANG, "layers": n_layers,
            "experts": n_exp, "top_k": top_k, "tokens": {}, "chars_per_token": {}, "articles": {}}

    for li, lang in enumerate(langs):
        t1 = time.time()
        ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
        seqs, chars, articles = [], 0, 0
        for row in ds:
            articles += 1
            text = row["text"].strip()
            if len(text) < 1500:
                continue
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) < SEQ_LEN:
                continue
            seqs.append(ids[:SEQ_LEN])
            chars += len(tok.decode(ids[:SEQ_LEN]))
            if len(seqs) >= SEQS_PER_LANG:
                break
        x = torch.tensor(seqs, device="cuda")
        half = len(seqs) // 2
        for b in range(0, len(seqs), BATCH):
            split = 0 if b < half else 1
            with torch.no_grad():
                out = model(input_ids=x[b:b + BATCH], output_router_logits=True, use_cache=False)
            for layer, logits in enumerate(out.router_logits):
                top = logits.float().topk(top_k, dim=-1).indices  # (tokens, top_k)
                counts[split, li, layer] += torch.bincount(top.flatten(), minlength=n_exp).cpu().numpy()
                if split == 0:
                    hot = torch.zeros(top.shape[0], n_exp, device="cuda")
                    hot.scatter_(1, top, 1.0)
                    cooc[li, layer] += (hot.T @ hot).round().to(torch.int32).cpu().numpy()
        meta["tokens"][lang] = int(x.numel())
        meta["chars_per_token"][lang] = round(chars / max(1, x.numel()), 3)
        meta["articles"][lang] = articles
        print(f"{lang}: {len(seqs)} sequences, {x.numel()} tokens, {meta['chars_per_token'][lang]} chars/token, "
              f"{time.time() - t1:.0f} s", flush=True)

    buf = io.BytesIO()
    np.savez_compressed(buf, counts=counts, cooc=cooc, meta=json.dumps(meta))
    data = buf.getvalue()
    pathlib.Path("/vol/results").mkdir(parents=True, exist_ok=True)
    pathlib.Path("/vol/results/routing_qwen3_30b_a3b.npz").write_bytes(data)
    vol.commit()
    print(f"done after {time.time() - t0:.0f} s; {len(data) / 1e6:.1f} MB", flush=True)
    return data


@app.local_entrypoint()
def main():
    data = run.remote()
    out = pathlib.Path(__file__).resolve().parent / "results" / "routing_qwen3_30b_a3b.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print("saved", out)
