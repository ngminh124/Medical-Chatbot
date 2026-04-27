"""Router package — all API route modules."""

from . import admin, audio, auth, chat, documents, health, models, rag

__all__ = [
	"audio",
	"admin",
	"auth",
	"chat",
	"documents",
	"health",
	"models",
	"rag",
]
