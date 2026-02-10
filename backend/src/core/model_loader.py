"""
Qwen3 Model Registry — Reranker + Guardrails only.

Embedding is served separately via HTTP (serving/qwen3_models/app.py on :7860),
so it is NOT loaded here.  This saves ~1.2 GB VRAM.

Models loaded locally (FP16 for deterministic VRAM budget):
  • Qwen3-Reranker-0.6B   – causal-LM yes/no scoring     (~1.2 GB FP16)
  • Qwen3Guard-Gen-0.6B   – 3-tier safety classification  (~1.2 GB FP16)
  Total local VRAM ≈ 2.4 GB (down from 4.8 GB previously)

References:
  - Qwen3-Reranker : https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
  - Qwen3Guard     : https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model_config import get_guardrails_model, get_reranking_model


class ModelRegistry:
    """Singleton registry for Qwen3 Reranker + Guardrails models."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Qwen3-Reranker-0.6B
        self.rerank_model = None
        self.rerank_tokenizer = None

        # Qwen3Guard-Gen-0.6B
        self.guard_model = None
        self.guard_tokenizer = None

        self._initialized = True
        logger.info(f"🔧 ModelRegistry initialized (device={self.device})")

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def load_models(self):
        """Load Reranker + Guardrails at startup (Embedding is external)."""
        logger.info("📦 Loading Qwen3 local models (Reranker + Guard)…")

        try:
            self._load_reranker()
            self._load_guardrails()

            logger.success("🎉 All local Qwen3 models loaded successfully!")
            self._warmup_models()

        except Exception as e:
            logger.error(f"❌ Failed to load Qwen3 models: {e}")
            raise

    # ── Reranker ────────────────────────────────────────────────────────

    def _load_reranker(self):
        rerank_repo = get_reranking_model()
        logger.info(f"Loading Qwen3-Reranker: {rerank_repo}")

        self.rerank_tokenizer = AutoTokenizer.from_pretrained(
            rerank_repo, padding_side="left"
        )
        self.rerank_model = AutoModelForCausalLM.from_pretrained(
            rerank_repo,
            trust_remote_code=True,
            torch_dtype=torch.float16,          # ← explicit FP16 (saves ~50 % VRAM vs FP32)
        ).to(self.device)
        self.rerank_model.eval()

        # Precompute yes/no token IDs (official Qwen3-Reranker spec)
        self.token_true_id = self.rerank_tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.rerank_tokenizer.convert_tokens_to_ids("no")

        # Precompute prefix / suffix tokens (official chat template)
        prefix = (
            '<|im_start|>system\n'
            'Judge whether the Document meets the requirements based on the '
            'Query and the Instruct provided. Note that the answer can only '
            'be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.rerank_prefix_tokens = self.rerank_tokenizer.encode(
            prefix, add_special_tokens=False
        )
        self.rerank_suffix_tokens = self.rerank_tokenizer.encode(
            suffix, add_special_tokens=False
        )
        self.rerank_max_length = 1024

        logger.info(
            f"✅ Qwen3-Reranker loaded — FP16, yes/no scoring, "
            f"max_len={self.rerank_max_length}"
        )

    # ── Guardrails ──────────────────────────────────────────────────────

    def _load_guardrails(self):
        guard_repo = get_guardrails_model()
        logger.info(f"Loading Qwen3Guard: {guard_repo}")

        self.guard_tokenizer = AutoTokenizer.from_pretrained(guard_repo)
        self.guard_model = AutoModelForCausalLM.from_pretrained(
            guard_repo,
            trust_remote_code=True,
            torch_dtype=torch.float16,          # ← explicit FP16
        ).to(self.device)
        self.guard_model.eval()

        logger.info("✅ Qwen3Guard loaded — FP16, 3-tier severity, 9 categories")

    # ── Warm-up ─────────────────────────────────────────────────────────

    def _warmup_models(self):
        """
        Run a single dummy inference through each local model so the first
        real request doesn't pay an initialisation penalty.
        """
        try:
            logger.info("🔥 Warming up Reranker & Guardrails…")

            # Reranker warm-up
            _ = self.rerank_documents(
                query="Warm-up query",
                documents=["Warm-up document."],
                top_n=1,
            )
            logger.debug("✅ Reranker warmed up")

            # Guardrails warm-up
            _ = self.check_safety(text="Hello", check_type="input")
            logger.debug("✅ Guardrails warmed up")

            logger.success("🔥 All local models warmed up!")

        except Exception as e:
            logger.warning(f"⚠️  Warm-up failed (non-critical): {e}")

    # ── Readiness ───────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """True when both local models are loaded and ready."""
        return self.rerank_model is not None and self.guard_model is not None

    # ================================================================== #
    #  Qwen3-Reranker                                                     #
    # ================================================================== #

    def rerank_documents(
        self,
        query: str,
        documents: List[str],
        top_n: int = 5,
        instruction: str = "Given a medical query, determine if the passage contains the answer",
    ) -> Tuple[List[float], List[int]]:
        """
        Rerank *documents* against *query* using Qwen3-Reranker-0.6B.

        Returns:
            (sorted_scores, sorted_indices) — descending by relevance.

        Official spec:
          • Format : "<Instruct>: …\n<Query>: …\n<Document>: …"
          • Scoring: logits[:, -1, :] → P(yes) via log-softmax
          • Ref    : https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
        """
        if self.rerank_model is None or self.rerank_tokenizer is None:
            raise RuntimeError("Qwen3-Reranker model not loaded")

        # Build formatted pairs ------------------------------------------------
        formatted_pairs = [
            f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
            for doc in documents
        ]

        # Tokenize (no padding yet — we'll add prefix/suffix first) ------------
        inputs = self.rerank_tokenizer(
            formatted_pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=(
                self.rerank_max_length
                - len(self.rerank_prefix_tokens)
                - len(self.rerank_suffix_tokens)
            ),
        )

        # Wrap with prefix / suffix tokens ------------------------------------
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = (
                self.rerank_prefix_tokens + ids + self.rerank_suffix_tokens
            )

        # Pad → tensors → device -----------------------------------------------
        inputs = self.rerank_tokenizer.pad(
            inputs, padding=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass — yes/no token scoring ----------------------------------
        with torch.no_grad():
            logits = self.rerank_model(**inputs).logits[:, -1, :]

            true_logits = logits[:, self.token_true_id]
            false_logits = logits[:, self.token_false_id]

            pair = torch.stack([false_logits, true_logits], dim=1)
            log_probs = torch.nn.functional.log_softmax(pair, dim=1)
            scores = log_probs[:, 1].exp().cpu().tolist()   # P(yes)

        # Sort descending by score --------------------------------------------
        sorted_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_n]
        sorted_scores = [scores[i] for i in sorted_indices]

        return sorted_scores, sorted_indices

    # ================================================================== #
    #  Qwen3Guard                                                         #
    # ================================================================== #

    @staticmethod
    def _parse_qwen3guard_output(text: str) -> Dict[str, Any]:
        """
        Parse the structured output produced by Qwen3Guard-Gen-0.6B.

        Expected format::

            Safety: Safe|Unsafe|Controversial
            Categories: <comma-separated>
            Refusal: Yes|No
        """
        result: Dict[str, Any] = {
            "severity": "Safe",
            "categories": [],
            "is_refusal": False,
        }

        severity_m = re.search(
            r"Safety:\s*(Safe|Unsafe|Controversial)", text, re.IGNORECASE
        )
        if severity_m:
            result["severity"] = severity_m.group(1).capitalize()

        cats_m = re.search(
            r"Categories:\s*(.+?)(?:\n|Refusal:|$)", text, re.IGNORECASE | re.DOTALL
        )
        if cats_m:
            raw = re.split(r"[,\n|]+", cats_m.group(1).strip())
            result["categories"] = [
                c.strip() for c in raw if c.strip() and c.strip() != "None"
            ]

        refusal_m = re.search(r"Refusal:\s*(Yes|No)", text, re.IGNORECASE)
        if refusal_m:
            result["is_refusal"] = refusal_m.group(1).lower() == "yes"

        return result

    def check_safety(
        self,
        text: str,
        check_type: str = "input",
        query: Optional[str] = None,
    ) -> Tuple[bool, str, List[str], bool, str]:
        """
        Evaluate content safety with Qwen3Guard-Gen-0.6B.

        Args:
            text:       Content to moderate (query **or** response).
            check_type: ``"input"`` for user queries,
                        ``"output"`` for LLM responses.
            query:      Original user query (required when *check_type="output"*).

        Returns:
            ``(is_safe, severity, categories, is_refusal, raw_output)``
        """
        if self.guard_model is None or self.guard_tokenizer is None:
            raise RuntimeError("Qwen3Guard model not loaded")

        # Build chat messages --------------------------------------------------
        if check_type == "input":
            messages = [{"role": "user", "content": text}]
        else:
            if not query:
                raise ValueError(
                    "query is REQUIRED for output moderation (check_type='output')"
                )
            messages = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": text},
            ]

        # Tokenize with chat template ------------------------------------------
        prompt = self.guard_tokenizer.apply_chat_template(
            messages, tokenize=False
        )
        model_inputs = self.guard_tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Generate (deterministic — do_sample=False for stable output) ---------
        with torch.no_grad():
            generated_ids = self.guard_model.generate(
                **model_inputs,
                max_new_tokens=64,      # ← reduced from 128 (faster)
                do_sample=False,        # ← deterministic decoding
            )

        # Decode only the NEW tokens -------------------------------------------
        new_ids = generated_ids[0][model_inputs["input_ids"].shape[1] :]
        output_text = self.guard_tokenizer.decode(
            new_ids, skip_special_tokens=True
        )

        parsed = self._parse_qwen3guard_output(output_text)
        is_safe = parsed["severity"] == "Safe"

        logger.debug(
            f"Guard result: severity={parsed['severity']}, "
            f"categories={parsed['categories']}, refusal={parsed['is_refusal']}"
        )

        return (
            is_safe,
            parsed["severity"],
            parsed["categories"],
            parsed["is_refusal"],
            output_text,
        )


# ══════════════════════════════════════════════════════════════════════ #
#  Module-level accessor                                                #
# ══════════════════════════════════════════════════════════════════════ #

_model_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Return the singleton :class:`ModelRegistry` instance."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry