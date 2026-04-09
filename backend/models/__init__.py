"""
backend.models — SQLAlchemy ORM models package.

Exports
-------
- Base          : declarative base (for create_all / metadata)
- User          : users table
- Thread        : threads table
- Message       : messages table
- Feedback      : feedbacks table
- Document      : documents table
- Chunk         : chunks table
- init_db()     : create all tables via engine
"""

from .base import Base
from .chat import Feedback, Message, Thread
from .knowledge import Chunk, Document
from .user import User

__all__ = [
    "Base",
    "User",
    "Thread",
    "Message",
    "Feedback",
    "Document",
    "Chunk",
    "init_db",
]


def init_db() -> None:
    """Create all tables that don't yet exist in the database.

    Uses the engine from ``backend.src.database`` and the metadata
    registered on ``Base``.
    """
    from loguru import logger

    from ..src.database import engine

    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables verified / created.")
