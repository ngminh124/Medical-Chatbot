"""Runtime-tunable settings for admin dashboard controls."""

from threading import Lock

from ..configs.setup import get_backend_settings

_settings = get_backend_settings()
_lock = Lock()

_runtime_settings = {
    "rewrite_enabled": True,
    "rerank_enabled": True,
    "max_tokens": 1024,
    "top_k": int(_settings.top_k),
}


def get_runtime_settings() -> dict:
    with _lock:
        return dict(_runtime_settings)


def update_runtime_settings(payload: dict) -> dict:
    allowed = {
        "rewrite_enabled",
        "rerank_enabled",
        "max_tokens",
        "top_k",
    }
    with _lock:
        for key, value in payload.items():
            if key not in allowed:
                continue
            _runtime_settings[key] = value
        return dict(_runtime_settings)
