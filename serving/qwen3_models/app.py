import os
import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
import numpy as np

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Configuration via environment variables ───────────────────────────
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B"
)
RERANKER_MODEL_NAME = os.environ.get(
    "RERANKER_MODEL_NAME", "Qwen/Qwen3-Reranker-0.6B"
)
GUARD_MODEL_NAME = os.environ.get(
    "GUARD_MODEL_NAME", "Qwen/Qwen3Guard-Gen-0.6B"
)
DEVICE = os.environ.get("DEVICE", "cuda")
BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "64"))


# ── Request / Response schemas ────────────────────────────────────────

class EmbedRequest(BaseModel):
    texts: List[str]
    normalize: Optional[bool] = True
    is_query: Optional[bool] = False
    instruction: Optional[str] = None
    batch_size: Optional[int] = None


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]


class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_n: int = 5
    instruction: Optional[str] = None


class RerankResponse(BaseModel):
    scores: List[float]
    indices: List[int]


class GuardRequest(BaseModel):
    text: str
    check_type: str = "input"          # "input" or "output"
    query: Optional[str] = None        # required when check_type="output"


class GuardResponse(BaseModel):
    raw_output: str
    severity: str
    categories: List[str]
    is_safe: bool


app = FastAPI(title="Qwen3 GPU Service (Embed · Rerank · Guard)")


def choose_device(requested: str) -> str:
    req = (requested or "cpu").lower()
    if req == "cuda" and torch.cuda.is_available():
        return "cuda"
    if req == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = choose_device(DEVICE)
logger.info(f"🚀 Qwen3 GPU Service starting — Device={DEVICE}")


# ══════════════════════════════════════════════════════════════════════
#  Model Loading (all models loaded at startup)
# ══════════════════════════════════════════════════════════════════════

# ── 1. Embedding (SentenceTransformer) ────────────────────────────────
try:
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
    embed_model.eval()
    EMBEDDING_DIM = embed_model.get_sentence_embedding_dimension()
    logger.success(
        f"✅ Embedding loaded: {EMBEDDING_MODEL_NAME} (dim={EMBEDDING_DIM})"
    )
except Exception as e:
    logger.error(f"❌ Failed to load embedding model: {e}")
    raise


# ── 2. Reranker (Qwen3-Reranker-0.6B, FP16) ─────────────────────────
rerank_model = None
rerank_tokenizer = None
token_true_id = None
token_false_id = None
rerank_prefix_tokens: List[int] = []
rerank_suffix_tokens: List[int] = []
RERANK_MAX_LENGTH = 1024

try:
    rerank_tokenizer = AutoTokenizer.from_pretrained(
        RERANKER_MODEL_NAME, padding_side="left"
    )
    rerank_model = AutoModelForCausalLM.from_pretrained(
        RERANKER_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(DEVICE)
    rerank_model.eval()

    # Precompute yes/no token IDs (official Qwen3-Reranker spec)
    token_true_id = rerank_tokenizer.convert_tokens_to_ids("yes")
    token_false_id = rerank_tokenizer.convert_tokens_to_ids("no")

    # Precompute prefix/suffix tokens (official chat template)
    _prefix = (
        '<|im_start|>system\n'
        'Judge whether the Document meets the requirements based on the '
        'Query and the Instruct provided. Note that the answer can only '
        'be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    _suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    rerank_prefix_tokens = rerank_tokenizer.encode(
        _prefix, add_special_tokens=False
    )
    rerank_suffix_tokens = rerank_tokenizer.encode(
        _suffix, add_special_tokens=False
    )

    logger.success(
        f"✅ Reranker loaded: {RERANKER_MODEL_NAME} (FP16, yes/no scoring)"
    )
except Exception as e:
    logger.warning(f"⚠️  Reranker not available (non-critical): {e}")


# ── 3. Guard (Qwen3Guard-Gen-0.6B, FP16) ─────────────────────────────
guard_model = None
guard_tokenizer = None

try:
    guard_tokenizer = AutoTokenizer.from_pretrained(GUARD_MODEL_NAME)
    guard_model = AutoModelForCausalLM.from_pretrained(
        GUARD_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(DEVICE)
    guard_model.eval()

    logger.success(
        f"✅ Guard loaded: {GUARD_MODEL_NAME} (FP16, 3-tier severity)"
    )
except Exception as e:
    logger.warning(f"⚠️  Guard not available (non-critical): {e}")


# ══════════════════════════════════════════════════════════════════════
#  Utility functions
# ══════════════════════════════════════════════════════════════════════

def l2_normalize(array: np.ndarray) -> np.ndarray:
    """L2-normalize embedding vectors."""
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def parse_guard_output(text: str) -> Dict[str, Any]:
    """Parse structured output produced by Qwen3Guard-Gen-0.6B."""
    result: Dict[str, Any] = {
        "severity": "Safe",
        "categories": ["None"],
        "is_refusal": False,
    }

    severity_m = re.search(
        r"Safety:\s*(Safe|Unsafe|Controversial)", text, re.IGNORECASE
    )
    if severity_m:
        result["severity"] = severity_m.group(1).capitalize()

    cats_m = re.search(
        r"Categories?:\s*(.+?)(?:\n|Refusal:|$)", text, re.IGNORECASE | re.DOTALL
    )
    if cats_m:
        raw = re.split(r"[,\n|]+", cats_m.group(1).strip())
        parsed_cats = [c.strip() for c in raw if c.strip()]
        if parsed_cats:
            result["categories"] = parsed_cats

    refusal_m = re.search(r"Refusal:\s*(Yes|No)", text, re.IGNORECASE)
    if refusal_m:
        result["is_refusal"] = refusal_m.group(1).lower() == "yes"

    return result


# ══════════════════════════════════════════════════════════════════════
#  Endpoints
# ══════════════════════════════════════════════════════════════════════

@app.get("/v1/ready")
def ready():
    """Health-check endpoint reporting readiness of each model."""
    return {
        "ready": True,
        "models": {
            "embedding": embed_model is not None,
            "reranker": rerank_model is not None,
            "guard": guard_model is not None,
        },
    }


# ── Embedding ─────────────────────────────────────────────────────────

@app.post("/v1/models/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    """Generate embeddings using Qwen3-Embedding-0.6B."""
    texts = req.texts or []
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise HTTPException(status_code=400, detail="`texts` must be a list of strings")

    batch_size = req.batch_size or BATCH_SIZE
    embeddings_out: List[List[float]] = []

    try:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Qwen3-Embedding best practice: prepend instruction for queries only
            if req.is_query and req.instruction:
                batch_texts = [f"{req.instruction}: {t}" for t in batch_texts]

            emb = embed_model.encode(
                batch_texts, batch_size=batch_size, convert_to_numpy=True
            )

            if req.normalize:
                emb = l2_normalize(emb)

            embeddings_out.extend(emb.tolist())

        return {"embeddings": embeddings_out}

    except Exception as e:
        logger.error(f"Error during embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Reranking ─────────────────────────────────────────────────────────

@app.post("/v1/models/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    """
    Rerank documents against a query using Qwen3-Reranker-0.6B.

    Official spec (https://huggingface.co/Qwen/Qwen3-Reranker-0.6B):
      • Format : <Instruct>: …\\n<Query>: …\\n<Document>: …
      • Scoring: logits[:, -1, :] → P(yes) via log-softmax over yes/no tokens
    """
    if rerank_model is None or rerank_tokenizer is None:
        raise HTTPException(status_code=503, detail="Reranker model not loaded")

    if not req.documents:
        return RerankResponse(scores=[], indices=[])

    instruction = req.instruction or (
        "Given a medical query, determine if the passage contains the answer"
    )

    try:
        # Build formatted pairs ────────────────────────────────────────
        formatted_pairs = [
            f"<Instruct>: {instruction}\n<Query>: {req.query}\n<Document>: {doc}"
            for doc in req.documents
        ]

        # Tokenize (without prefix/suffix yet) ─────────────────────────
        inputs = rerank_tokenizer(
            formatted_pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=(
                RERANK_MAX_LENGTH
                - len(rerank_prefix_tokens)
                - len(rerank_suffix_tokens)
            ),
        )

        # Wrap with prefix / suffix tokens ─────────────────────────────
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = (
                rerank_prefix_tokens + ids + rerank_suffix_tokens
            )

        # Pad → tensors → device ───────────────────────────────────────
        inputs = rerank_tokenizer.pad(
            inputs, padding=True, return_tensors="pt"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Forward pass — yes/no token scoring ──────────────────────────
        with torch.no_grad():
            logits = rerank_model(**inputs).logits[:, -1, :]

            true_logits = logits[:, token_true_id]
            false_logits = logits[:, token_false_id]

            pair = torch.stack([false_logits, true_logits], dim=1)
            log_probs = torch.nn.functional.log_softmax(pair, dim=1)
            scores = log_probs[:, 1].exp().cpu().tolist()   # P(yes)

        # Sort descending by score ─────────────────────────────────────
        sorted_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[: req.top_n]
        sorted_scores = [scores[i] for i in sorted_indices]

        return RerankResponse(scores=sorted_scores, indices=sorted_indices)

    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Guard (content moderation) ────────────────────────────────────────

@app.post("/v1/models/guard", response_model=GuardResponse)
def guard(req: GuardRequest):
    """
    Check content safety using Qwen3Guard-Gen-0.6B.

    Official spec (https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B):
      • Three-tiered severity : Safe · Unsafe · Controversial
      • 9 safety categories
      • Output: Safety: {label}\\nCategories: {cats}\\nRefusal: {yes/no}
    """
    if guard_model is None or guard_tokenizer is None:
        raise HTTPException(status_code=503, detail="Guard model not loaded")

    try:
        # Build chat messages ──────────────────────────────────────────
        if req.check_type == "input":
            messages = [{"role": "user", "content": req.text}]
        else:
            if not req.query:
                raise HTTPException(
                    status_code=400,
                    detail="query is required for output moderation "
                           "(check_type='output')",
                )
            messages = [
                {"role": "user", "content": req.query},
                {"role": "assistant", "content": req.text},
            ]

        # Tokenize with chat template ─────────────────────────────────
        prompt = guard_tokenizer.apply_chat_template(
            messages, tokenize=False
        )
        model_inputs = guard_tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(DEVICE)

        # Generate (deterministic — do_sample=False) ───────────────────
        with torch.no_grad():
            generated_ids = guard_model.generate(
                **model_inputs,
                max_new_tokens=64,
                do_sample=False,
            )

        # Decode only the NEW tokens ──────────────────────────────────
        new_ids = generated_ids[0][model_inputs["input_ids"].shape[1] :]
        output_text = guard_tokenizer.decode(
            new_ids, skip_special_tokens=True
        )

        # Parse structured output ──────────────────────────────────────
        parsed = parse_guard_output(output_text)

        return GuardResponse(
            raw_output=output_text,
            severity=parsed["severity"],
            categories=parsed["categories"],
            is_safe=parsed["severity"] == "Safe",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during guard check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
