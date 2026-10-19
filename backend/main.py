from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import json
import os

app = FastAPI(title="Transformer Explainer API")

# CORS so frontend can access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Model Once At Startup ───────────────────────────────────────────────
print("Loading GPT-2 model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained(
    "gpt2",
    output_attentions=True,
    output_hidden_states=True,
)
model.eval()
print("Model loaded successfully!")

# GPT-2 meta
GPT2_META = {
    "layer_num": 12,
    "attention_head_num": 12,
    "dimension": 768,
    "vocab_size": 50257,
}

# ─── Request / Response Models ─────────────────────────────────────────────────
class InferRequest(BaseModel):
    text: str
    temperature: float = 0.8
    sampling_type: str = "top-k"   # "top-k" or "top-p"
    sampling_value: float = 5.0    # k or p value

# ─── Helper Functions ──────────────────────────────────────────────────────────
def softmax(logits):
    logits = np.array(logits, dtype=np.float64)
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    return exp_logits / np.sum(exp_logits)

def top_k_sampling(logits_np, token_ids_list, k=5, temperature=0.8):
    max_display = 50
    sorted_indices = np.argsort(logits_np)[::-1][:max_display]
    sorted_logits = logits_np[sorted_indices]

    # Temperature scaling
    scaled = sorted_logits / temperature

    # Top-k masking
    filtered = np.full_like(scaled, -np.inf)
    k = min(int(k), max_display)
    filtered[:k] = scaled[:k]

    exp_logits = np.exp(filtered - np.max(filtered[np.isfinite(filtered)]))
    exp_logits[~np.isfinite(filtered)] = 0.0
    probs = exp_logits / np.sum(exp_logits)

    probabilities = []
    for rank, (idx, logit, scaled_l, exp_l, prob) in enumerate(
        zip(sorted_indices, sorted_logits, scaled, exp_logits, probs)
    ):
        token_str = tokenizer.decode([int(idx)])
        token_str = format_token(token_str)
        top_k_logit = float(filtered[rank]) if np.isfinite(filtered[rank]) else None
        probabilities.append({
            "tokenId": int(idx),
            "token": token_str,
            "logit": float(logit),
            "scaledLogit": float(scaled_l),
            "topKLogit": top_k_logit,
            "expLogit": float(exp_l),
            "probability": float(prob),
            "rank": rank,
        })

    # Sample
    valid_probs = np.array([p["probability"] for p in probabilities])
    if valid_probs.sum() == 0:
        valid_probs = np.ones(len(probabilities)) / len(probabilities)
    sampled_idx = np.random.choice(len(probabilities), p=valid_probs / valid_probs.sum())
    sampled = probabilities[sampled_idx]

    return probabilities, sampled

def top_p_sampling(logits_np, k=50, p=0.9, temperature=0.8):
    sorted_indices = np.argsort(logits_np)[::-1][:k]
    sorted_logits = logits_np[sorted_indices]
    scaled = sorted_logits / temperature

    exp_l = np.exp(scaled - np.max(scaled))
    probs = exp_l / np.sum(exp_l)
    cumulative = np.cumsum(probs)

    cutoff = int(np.searchsorted(cumulative, p))
    cutoff = max(cutoff, 0)
    top_probs = probs[:cutoff + 1]
    top_probs_renorm = top_probs / top_probs.sum() if top_probs.sum() > 0 else top_probs

    probabilities = []
    for rank, (idx, logit, sc, ep, prob, cum) in enumerate(
        zip(sorted_indices, sorted_logits, scaled, exp_l, probs, cumulative)
    ):
        token_str = format_token(tokenizer.decode([int(idx)]))
        final_prob = float(top_probs_renorm[rank]) if rank <= cutoff else 0.0
        probabilities.append({
            "tokenId": int(idx),
            "token": token_str,
            "logit": float(logit),
            "scaledLogit": float(sc),
            "expLogit": float(ep),
            "topPProbability": float(prob),
            "cumulativeProbability": float(cum),
            "probability": final_prob,
            "cutoffIndex": cutoff,
            "rank": rank,
        })

    valid_probs = np.array([p["probability"] for p in probabilities])
    if valid_probs.sum() == 0:
        sampled = probabilities[0]
    else:
        si = np.random.choice(len(probabilities), p=valid_probs / valid_probs.sum())
        sampled = probabilities[si]

    return probabilities, sampled

def format_token(token: str) -> str:
    return (
        token.replace("\n", "[NEWLINE]")
             .replace("\t", "[TAB]")
             .replace("\r", "[CR]")
    )

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "model": "gpt2"}

@app.get("/api/meta")
def meta():
    return GPT2_META

@app.post("/api/infer")
def infer(req: InferRequest):
    text = req.text.strip() or " "

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    token_id_list = input_ids[0].tolist()
    token_strings = [format_token(tokenizer.decode([tid])) for tid in token_id_list]

    if len(token_id_list) > 1024:
        raise HTTPException(status_code=400, detail="Input too long (max 1024 tokens)")

    with torch.no_grad():
        outputs = model(
            input_ids,
            output_attentions=True,
            output_hidden_states=True,
        )

    # Logits for last token
    logits = outputs.logits[0, -1, :].numpy()  # shape: (vocab_size,)

    # Attention weights — shape per layer: (1, heads, seq, seq)
    attentions = []
    for layer_idx, layer_attn in enumerate(outputs.attentions):
        layer_data = []
        attn_np = layer_attn[0].numpy()  # (heads, seq, seq)
        for head_idx in range(GPT2_META["attention_head_num"]):
            head_attn = attn_np[head_idx].tolist()  # (seq, seq)
            layer_data.append(head_attn)
        attentions.append(layer_data)

    # Hidden states — list of (1, seq, 768), one per layer + embedding
    hidden_states = []
    for hs in outputs.hidden_states:
        hidden_states.append(hs[0].numpy().tolist())  # (seq, 768)

    # Sampling
    if req.sampling_type == "top-p":
        probabilities, sampled = top_p_sampling(
            logits, k=50, p=req.sampling_value, temperature=req.temperature
        )
    else:
        probabilities, sampled = top_k_sampling(
            logits, token_id_list, k=req.sampling_value, temperature=req.temperature
        )

    return {
        "tokens": token_strings,
        "tokenIds": token_id_list,
        "logits": logits.tolist(),
        "attentions": attentions,         # [layer][head][seq][seq]
        "hiddenStates": hidden_states,    # [layer][seq][768]
        "probabilities": probabilities,
        "sampled": sampled,
    }

@app.post("/api/tokenize")
def tokenize(req: BaseModel):
    pass

# Serve frontend
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
