import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
RERANKER_MODEL_NAME = os.environ.get("RERANKER_MODEL_NAME", "Qwen/Qwen3-Reranker-0.6B")
GUARD_MODEL_NAME = os.environ.get("GUARD_MODEL_NAME", "Qwen/Qwen3Guard-Gen-0.6B")

REQUESTED_EMBED_DEVICE = os.environ.get("DEVICE", "cuda")

EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))
EMBED_MAX_BATCH_SIZE = int(os.environ.get("EMBED_MAX_BATCH_SIZE", "128"))
RERANK_MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "1024"))
GUARD_MAX_LENGTH = int(os.environ.get("GUARD_MAX_LENGTH", "512"))
GUARD_MAX_NEW_TOKENS = int(os.environ.get("GUARD_MAX_NEW_TOKENS", "64"))


def _choose_embedding_device(requested: str) -> str:
    req = (requested or "cpu").lower()
    if req == "cuda" and torch.cuda.is_available():
        return "cuda"
    if req == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


EMBEDDING_DEVICE = _choose_embedding_device(REQUESTED_EMBED_DEVICE)
logger.info(
    f"Service init (lazy): embedding_device={EMBEDDING_DEVICE}, rerank_device=cpu, guard_device=cpu"
)


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
    check_type: str = "input"
    query: Optional[str] = None


class GuardResponse(BaseModel):
    raw_output: str
    severity: str
    categories: List[str]
    is_safe: bool


app = FastAPI(title="Qwen3 Model Service (Lazy · Stable)")


def l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def parse_guard_output(text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "severity": "Safe",
        "categories": ["None"],
        "is_refusal": False,
    }

    severity_m = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", text, re.IGNORECASE)
    if severity_m:
        result["severity"] = severity_m.group(1).capitalize()

    cats_m = re.search(r"Categories?:\s*(.+?)(?:\n|Refusal:|$)", text, re.IGNORECASE | re.DOTALL)
    if cats_m:
        raw = re.split(r"[,\n|]+", cats_m.group(1).strip())
        parsed = [c.strip() for c in raw if c.strip()]
        if parsed:
            result["categories"] = parsed

    refusal_m = re.search(r"Refusal:\s*(Yes|No)", text, re.IGNORECASE)
    if refusal_m:
        result["is_refusal"] = refusal_m.group(1).lower() == "yes"

    return result


@dataclass
class RerankBundle:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    token_true_id: int
    token_false_id: int
    prefix_tokens: List[int]
    suffix_tokens: List[int]


@dataclass
class GuardBundle:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM


class ModelManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._embed_model: Optional[SentenceTransformer] = None
        self._embed_device: str = EMBEDDING_DEVICE
        self._embed_dim: Optional[int] = None

        self._rerank_bundle: Optional[RerankBundle] = None
        self._guard_bundle: Optional[GuardBundle] = None

    def get_embedding_model(self) -> Tuple[SentenceTransformer, str, int]:
        with self._lock:
            if self._embed_model is None:
                logger.info(f"[LOAD] Embedding -> {EMBEDDING_MODEL_NAME} on {self._embed_device}")
                self._embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self._embed_device)
                self._embed_model.eval()
                self._embed_dim = int(self._embed_model.get_sentence_embedding_dimension())
                logger.success(f"[LOAD] Embedding ready (dim={self._embed_dim}, device={self._embed_device})")

            return self._embed_model, self._embed_device, int(self._embed_dim or 0)

    def fallback_embedding_to_cpu(self):
        with self._lock:
            if self._embed_device == "cpu":
                return
            logger.warning("[EMBED] Fallback to CPU due to GPU instability/OOM")
            self._embed_model = None
            self._embed_dim = None
            self._embed_device = "cpu"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_rerank_bundle(self) -> RerankBundle:
        with self._lock:
            if self._rerank_bundle is None:
                logger.info(f"[LOAD] Reranker -> {RERANKER_MODEL_NAME} on cpu")
                tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME, padding_side="left")
                model = AutoModelForCausalLM.from_pretrained(
                    RERANKER_MODEL_NAME,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                ).to("cpu")
                model.eval()

                token_true_id = tokenizer.convert_tokens_to_ids("yes")
                token_false_id = tokenizer.convert_tokens_to_ids("no")
                if token_true_id is None or token_true_id < 0 or token_false_id is None or token_false_id < 0:
                    raise RuntimeError("Cannot resolve 'yes'/'no' token ids for reranker")

                prefix = (
                    '<|im_start|>system\n'
                    'Judge whether the Document meets the requirements based on the '
                    'Query and the Instruct provided. Note that the answer can only '
                    'be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
                )
                suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
                suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

                self._rerank_bundle = RerankBundle(
                    tokenizer=tokenizer,
                    model=model,
                    token_true_id=int(token_true_id),
                    token_false_id=int(token_false_id),
                    prefix_tokens=prefix_tokens,
                    suffix_tokens=suffix_tokens,
                )
                logger.success("[LOAD] Reranker ready on CPU")

            return self._rerank_bundle

    def get_guard_bundle(self) -> GuardBundle:
        with self._lock:
            if self._guard_bundle is None:
                logger.info(f"[LOAD] Guard -> {GUARD_MODEL_NAME} on cpu")
                tokenizer = AutoTokenizer.from_pretrained(GUARD_MODEL_NAME)
                model = AutoModelForCausalLM.from_pretrained(
                    GUARD_MODEL_NAME,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                ).to("cpu")
                model.eval()
                self._guard_bundle = GuardBundle(tokenizer=tokenizer, model=model)
                logger.success("[LOAD] Guard ready on CPU")

            return self._guard_bundle

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "embedding": {
                    "loaded": self._embed_model is not None,
                    "device": self._embed_device,
                    "dim": self._embed_dim,
                },
                "reranker": {
                    "loaded": self._rerank_bundle is not None,
                    "device": "cpu",
                },
                "guard": {
                    "loaded": self._guard_bundle is not None,
                    "device": "cpu",
                },
            }


manager = ModelManager()


@app.get("/v1/health")
def health():
    cuda_info = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        try:
            cuda_info["allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 / 1024, 2)
            cuda_info["reserved_mb"] = round(torch.cuda.memory_reserved() / 1024 / 1024, 2)
        except Exception:
            pass

    return {
        "ok": True,
        "models": manager.status(),
        "cuda": cuda_info,
    }


@app.get("/v1/ready")
def ready():
    status = manager.status()
    return {
        "ready": True,
        "models": {
            "embedding": status["embedding"]["loaded"],
            "reranker": status["reranker"]["loaded"],
            "guard": status["guard"]["loaded"],
        },
        "devices": {
            "embedding": status["embedding"]["device"],
            "reranker": "cpu",
            "guard": "cpu",
        },
    }


def _embed_impl(req: EmbedRequest) -> EmbedResponse:
    texts = req.texts or []
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise HTTPException(status_code=400, detail="`texts` must be a list of strings")

    if not texts:
        return EmbedResponse(embeddings=[])

    batch_size = int(req.batch_size or EMBED_BATCH_SIZE)
    batch_size = max(1, min(batch_size, EMBED_MAX_BATCH_SIZE))

    model, device, _ = manager.get_embedding_model()

    outputs: List[List[float]] = []
    try:
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                if req.is_query and req.instruction:
                    batch = [f"{req.instruction}: {t}" for t in batch]

                emb = model.encode(batch, batch_size=batch_size, convert_to_numpy=True)
                if req.normalize:
                    emb = l2_normalize(emb)
                outputs.extend(emb.tolist())
        return EmbedResponse(embeddings=outputs)
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg and device == "cuda":
            logger.error("[EMBED] CUDA OOM detected; switching embedding model to CPU")
            manager.fallback_embedding_to_cpu()
            raise HTTPException(status_code=503, detail="GPU OOM. Embedding switched to CPU. Retry request.")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")


@app.post("/v1/models/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    return await run_in_threadpool(_embed_impl, req)


def _rerank_impl(req: RerankRequest) -> RerankResponse:
    if not req.documents:
        return RerankResponse(scores=[], indices=[])

    bundle = manager.get_rerank_bundle()

    instruction = req.instruction or "Given a medical query, determine if the passage contains the answer"
    top_n = max(1, min(req.top_n, len(req.documents)))

    formatted_pairs = [
        f"<Instruct>: {instruction}\n<Query>: {req.query}\n<Document>: {doc}"
        for doc in req.documents
    ]

    max_len = max(64, RERANK_MAX_LENGTH - len(bundle.prefix_tokens) - len(bundle.suffix_tokens))
    inputs = bundle.tokenizer(
        formatted_pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=max_len,
    )

    for i, ids in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = bundle.prefix_tokens + ids + bundle.suffix_tokens

    padded = bundle.tokenizer.pad(inputs, padding=True, return_tensors="pt")
    padded = {k: v.to("cpu") for k, v in padded.items()}

    with torch.inference_mode():
        logits = bundle.model(**padded).logits[:, -1, :]
        true_logits = logits[:, bundle.token_true_id]
        false_logits = logits[:, bundle.token_false_id]
        pair = torch.stack([false_logits, true_logits], dim=1)
        scores = torch.nn.functional.softmax(pair, dim=1)[:, 1].cpu().tolist()

    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    sorted_scores = [scores[i] for i in sorted_indices]
    return RerankResponse(scores=sorted_scores, indices=sorted_indices)


@app.post("/v1/models/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    try:
        return await run_in_threadpool(_rerank_impl, req)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rerank failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rerank failed: {e}")


def _guard_impl(req: GuardRequest) -> GuardResponse:
    bundle = manager.get_guard_bundle()

    if req.check_type not in {"input", "output"}:
        raise HTTPException(status_code=400, detail="check_type must be 'input' or 'output'")

    if req.check_type == "input":
        messages = [{"role": "user", "content": req.text}]
    else:
        if not req.query:
            raise HTTPException(status_code=400, detail="query is required when check_type='output'")
        messages = [
            {"role": "user", "content": req.query},
            {"role": "assistant", "content": req.text},
        ]

    prompt = bundle.tokenizer.apply_chat_template(messages, tokenize=False)
    model_inputs = bundle.tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=GUARD_MAX_LENGTH,
    ).to("cpu")

    with torch.inference_mode():
        generated_ids = bundle.model.generate(
            **model_inputs,
            max_new_tokens=GUARD_MAX_NEW_TOKENS,
            do_sample=False,
        )

    new_ids = generated_ids[0][model_inputs["input_ids"].shape[1] :]
    output_text = bundle.tokenizer.decode(new_ids, skip_special_tokens=True)
    parsed = parse_guard_output(output_text)

    return GuardResponse(
        raw_output=output_text,
        severity=parsed["severity"],
        categories=parsed["categories"],
        is_safe=parsed["severity"] == "Safe",
    )


@app.post("/v1/models/guard", response_model=GuardResponse)
async def guard(req: GuardRequest):
    try:
        return await run_in_threadpool(_guard_impl, req)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Guard failed: {e}")
        raise HTTPException(status_code=500, detail=f"Guard failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")