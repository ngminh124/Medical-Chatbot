import uuid

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from loguru import logger
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from .configs.celery_config import get_celery_app
from .configs.logging_config import get_rag_logger
from .configs.setup import get_backend_settings
from .core.guardrails import get_guardrails_service
from .core.vectorize import search_vectors, upsert_points
from .services.brain import detect_route, get_response
from .services.chunking import fixed_semantic_chunking
from .services.embedding import get_embedding_service
from .services.rerank import get_qwen3_reranker

settings = get_backend_settings()
tracer = trace.get_tracer(__name__)
rag_log = get_rag_logger()

celery_app = get_celery_app(__name__)
celery_app.autodiscover_tasks()


@shared_task
def chunk_and_index_document(doc_id, title, content, metadata=None):
    """
    Chunk document and index to both Qdrant (vector) and Elasticsearch (keyword).

    This implements dual indexing strategy (T088) for hybrid search:
    1. Chunk document using fixed semantic strategy
    2. Generate embeddings for each chunk
    3. Store chunks in PostgreSQL with enhanced metadata (T095)
    4. Index to Qdrant (vector search) with metadata (T097)
    5. Index to Elasticsearch (keyword search) with metadata (T098)

    Args:
        doc_id: Document ID
        title: Document title
        content: Document content
        metadata: Document metadata dict with keys:
            - source: Dataset source
            - doc_type: Document type (clinical_guideline, drug_info, medical_qa, etc.)
            - language: Language code (default: vi)
            - section_title: Section title (optional)
            - page_number: Page number (optional)
            - any other custom fields
    """
    try:
        from uuid import UUID

        from .database import SessionLocal
        from .models import Chunk

        metadata = metadata or {}

        # Chunk the document using fixed semantic strategy (T087)
        nodes = fixed_semantic_chunking(
            text=content, metadata={"doc_id": doc_id, "title": title}
        )

        # Get services
        embedding_service = get_embedding_service()
        from .services.elasticsearch import get_elasticsearch_client

        es_client = get_elasticsearch_client()

        # Generate embeddings and prepare points for dual indexing
        qdrant_points = []
        elasticsearch_docs = []
        db_chunks = []  # Store chunks in database (T095)

        # Extract metadata fields (T095)
        doc_source = metadata.get("source", "")
        doc_type = metadata.get("doc_type", "medical_qa")
        language = metadata.get("language", "vi")
        section_title = metadata.get("section_title")
        page_number = metadata.get("page_number")

        # OPTIMIZATION: Generate embeddings in batch for better performance
        chunk_texts = [node.text for node in nodes]
        batch_embeddings = embedding_service.embed_batch_documents(
            documents=chunk_texts,
            batch_size=512,  # Large batch size for dataset ingestion
        )

        for chunk_index, (node, embedding) in enumerate(zip(nodes, batch_embeddings)):
            if not embedding:
                continue

            # Generate unique chunk ID
            chunk_id = str(uuid.uuid4())

            # Calculate token count for metadata
            token_count = len(node.text.split())  # Rough estimate

            # Prepare enhanced metadata (T095: source_document_id, chunk_index, section_title, page_number)
            chunk_metadata = {
                "source_document_id": doc_id,
                "chunk_index": chunk_index,
                "title": title,
                "doc_type": doc_type,
                "source": doc_source,
                "language": language,
                "token_count": token_count,
            }

            # Add optional fields if present
            if section_title:
                chunk_metadata["section_title"] = section_title
            if page_number is not None:
                chunk_metadata["page_number"] = page_number

            # Add any additional metadata from document
            for key, value in metadata.items():
                if key not in [
                    "source",
                    "doc_type",
                    "language",
                    "section_title",
                    "page_number",
                ]:
                    chunk_metadata[key] = value

            # Prepare Qdrant point with enhanced metadata (T097)
            qdrant_point = {
                "id": chunk_id,
                "embedding": embedding,
                "metadata": {
                    "content": node.text,
                    **chunk_metadata,  # Include all enhanced metadata
                },
            }
            qdrant_points.append(qdrant_point)

            # Prepare Elasticsearch document with enhanced metadata (T098)
            elasticsearch_docs.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "chunk_index": chunk_index,
                    "content": node.text,
                    "title": title,
                    "doc_type": doc_type,
                    "source": doc_source,
                    "language": language,
                    "metadata": chunk_metadata,
                }
            )

            # Prepare database chunk (T095)
            db_chunks.append(
                {
                    "id": UUID(chunk_id),
                    "documentId": UUID(doc_id),
                    "chunkIndex": chunk_index,
                    "content": node.text,
                    "metadata_": chunk_metadata,
                }
            )

        # Store chunks in database (T095)
        if db_chunks:
            with SessionLocal() as db:
                for chunk_data in db_chunks:
                    chunk = Chunk(**chunk_data)
                    db.add(chunk)
                db.commit()

        # Index to Qdrant (vector database) with enhanced metadata (T097)
        if qdrant_points:
            upsert_points(
                qdrant_points, collection_name=settings.default_collection_name
            )

        # Index to Elasticsearch (keyword search) with enhanced metadata (T098)
        if elasticsearch_docs:
            for es_doc in elasticsearch_docs:
                es_client.index_chunk(
                    chunk_id=es_doc["chunk_id"],
                    document_id=es_doc["document_id"],
                    chunk_index=es_doc["chunk_index"],
                    content=es_doc["content"],
                    title=es_doc["title"],
                    doc_type=es_doc["doc_type"],
                    source=es_doc["source"],
                    language=es_doc["language"],
                    metadata=es_doc["metadata"],
                )

        return {"chunks_created": len(qdrant_points)}

    except Exception as e:
        logger.error(f"[CHUNK] ❌ Error indexing doc_id={doc_id}: {e}")
        raise


@shared_task()
def bot_route_answer_message(history, question, system_prompt=None):
    """
    Route user message to appropriate handler (medical RAG or general chat).

    Flow:
    1. Validate user input with Qwen3Guard (BEFORE routing)
    2. If invalid, return rejection message immediately
    3. If valid, detect route and proceed to appropriate handler

    Args:
        history: Conversation history
        question: User question
        system_prompt: Custom system prompt (optional, defaults to settings.system_prompt)

    Returns:
        str: Generated response
    """
    # ============================================
    # STEP 0: INPUT VALIDATION (Qwen3Guard) - BEFORE ROUTING
    # ============================================
    guardrails = get_guardrails_service()
    is_valid_input, violation_category, input_metadata = guardrails.validate_query(
        question
    )

    rag_log.log_guardrails_input(
        is_valid=is_valid_input,
        category=violation_category,
        severity=input_metadata.get("severity") if input_metadata else None,
    )

    if not is_valid_input:
        rejection_message = guardrails.get_rejection_message(
            violation_category, language="vi"
        )
        logger.warning(
            f"[GUARD] ⛔ Input rejected: category={violation_category}, "
            f"severity={input_metadata.get('severity') if input_metadata else 'unknown'}"
        )
        return rejection_message

    # ============================================
    # STEP 1: ROUTE DETECTION
    # ============================================
    route = detect_route(history, question)
    rag_log.log_route_detection(route)

    if route == "medical":
        # Pass skip_input_validation=True since we already validated
        return rag_qa_task(
            history, question, system_prompt=system_prompt, skip_input_validation=True
        )
    elif route == "general":
        response = get_response(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                    or "Bạn là Minqes - Trợ lý y tế thông minh cho người Việt.",
                }
            ]
            + history
            + [{"role": "user", "content": question}],
            temperature=0.3,
            max_tokens=512,
        )
        if not response:
            return "Xin lỗi, hệ thống đang quá tải. Vui lòng thử lại sau."
        return response


@shared_task(
    bind=True,
    time_limit=300,  # 5 minutes hard timeout
    soft_time_limit=180,  # 3 minutes soft timeout
    max_retries=0,  # No automatic retries
)
def rag_qa_task(
    self, history, question, system_prompt=None, skip_input_validation=False
):
    """RAG task delegates to the main RAG pipeline (single source of truth)."""
    import time

    request_start = time.time()
    try:
        from .src.routers.rag import run_rag_pipeline

        result = run_rag_pipeline(
            question=question,
            history=history or [],
            top_k=int(settings.top_k),
            web_search_enabled=False,
        )
        answer = (result or {}).get("answer") or "Xin lỗi, không thể tạo câu trả lời lúc này."
        rag_log.log_request_complete(request_start, success=True)
        return answer

    except SoftTimeLimitExceeded:
        logger.error("[RAG] Task exceeded soft time limit (180s)")
        rag_log.log_request_complete(request_start, success=False)
        return "Xin lỗi, yêu cầu của bạn đã vượt quá thời gian xử lý cho phép. Vui lòng thử lại với câu hỏi ngắn gọn hơn."

    except Exception as e:
        logger.error(f"[RAG] Error: {e}", exc_info=True)
        rag_log.log_request_complete(request_start, success=False)
        return "Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý câu hỏi."


@shared_task(bind=True)
def ingest_dataset_task(
    self,
    dataset_name: str,
    dataset_config: str = None,
    split: str = "train",
    doc_type: str = None,
    max_documents: int = None,
    batch_size: int = 10,  # NEW: Process documents in batches
):
    """
    Load a HuggingFace dataset and index all documents to Qdrant + Elasticsearch.

    OPTIMIZATIONS:
    - Batch database commits (reduce DB overhead)
    - Batch embedding generation (already in chunk_and_index_document)
    - Batch Elasticsearch indexing (via bulk API)

    Args:
        dataset_name: HuggingFace dataset identifier
        dataset_config: Dataset configuration name (optional)
        split: Dataset split to load (default: "train")
        doc_type: Document type for all documents
        max_documents: Limit number of documents (for testing)
        batch_size: Number of documents to process in each batch (default: 10)

    Returns:
        dict: {
            "documents_indexed": int,
            "chunks_indexed": int,
            "duration_seconds": float,
        }
    """
    import hashlib
    import time

    from datasets import load_dataset

    from .database import SessionLocal
    from .models import Chunk, Document

    start_time = time.time()
    documents_indexed = 0
    chunks_indexed = 0

    try:
        # Update state to running
        self.update_state(
            state="PROGRESS",
            meta={
                "documents_processed": 0,
                "total_documents": 0,
                "chunks_created": 0,
            },
        )

        # Load dataset from HuggingFace
        dataset = load_dataset(dataset_name, dataset_config, split=split)

        total_docs = (
            len(dataset) if max_documents is None else min(len(dataset), max_documents)
        )
        logger.info(
            f"[INDEX] 📂 Dataset loaded: {dataset_name} | {total_docs} documents | batch_size={batch_size}"
        )

        # OPTIMIZATION 1: Process documents in batches
        document_batch = []

        with SessionLocal() as db:
            for idx, item in enumerate(dataset):
                # CRITICAL FIX: Check max_documents against PROCESSED count, not dataset index
                if max_documents and documents_indexed >= max_documents:
                    break

                # Extract document fields (adapt to dataset structure)
                # Common field names: title, text, content, question, answer, etc.
                title = item.get("title") or item.get("question") or f"Document {idx}"
                content = (
                    item.get("text") or item.get("content") or item.get("answer") or ""
                )

                if not content:
                    continue

                # Calculate content hash for incremental updates (T102a)
                content_hash = hashlib.sha256(content.encode()).hexdigest()

                # Check if document already exists with same hash
                existing_doc = (
                    db.query(Document)
                    .filter(Document.metadata_["content_hash"].astext == content_hash)
                    .first()
                )

                if existing_doc:
                    continue

                # Create document metadata with version tracking (T102b)
                metadata = {
                    "source": dataset_name,
                    "doc_type": doc_type or "medical_qa",
                    "language": "vi",
                    "dataset_split": split,
                    "content_hash": content_hash,
                    "is_indexed": False,
                    "dataset_version": "1.0",  # T102b: Track dataset version for incremental updates
                    "indexed_at": None,  # Will be set when indexing completes
                }

                # Add any additional fields from dataset
                for key, value in item.items():
                    if key not in ["title", "text", "content", "question", "answer"]:
                        metadata[key] = value

                # Add to batch
                document_batch.append(
                    {
                        "title": title,
                        "content": content,
                        "metadata": metadata,
                        "idx": idx,
                    }
                )

                # OPTIMIZATION 2: Process batch when full or at end
                if len(document_batch) >= batch_size or idx == len(dataset) - 1:
                    # Create documents in database (batch commit)
                    new_docs = []
                    for doc_data in document_batch:
                        new_doc = Document(
                            title=doc_data["title"],
                            content=doc_data["content"],
                            metadata_=doc_data["metadata"],
                        )
                        db.add(new_doc)
                        new_docs.append((new_doc, doc_data))

                    # OPTIMIZATION 3: Single commit for batch
                    db.commit()

                    # Refresh all documents to get IDs
                    for new_doc, _ in new_docs:
                        db.refresh(new_doc)

                    # Chunk and index each document (embedding is already batched inside)
                    for new_doc, doc_data in new_docs:
                        chunk_result = chunk_and_index_document(
                            str(new_doc.id),
                            doc_data["title"],
                            doc_data["content"],
                            metadata=doc_data["metadata"],
                        )

                        # Update document as indexed with timestamp (T102b)
                        from datetime import datetime

                        new_doc.metadata_["is_indexed"] = True
                        new_doc.metadata_["indexed_at"] = datetime.utcnow().isoformat()

                        documents_indexed += 1
                        chunks_indexed += chunk_result.get("chunks_created", 0)

                    # OPTIMIZATION 4: Batch commit for status updates
                    db.commit()

                    # Update progress
                    self.update_state(
                        state="PROGRESS",
                        meta={
                            "documents_processed": documents_indexed,
                            "total_documents": total_docs,
                            "chunks_created": chunks_indexed,
                        },
                    )

                    # Log batch progress
                    logger.info(
                        f"[INDEX] 📊 Progress: {documents_indexed}/{total_docs} docs | {chunks_indexed} chunks"
                    )

                    # Clear batch
                    document_batch = []

        duration = time.time() - start_time

        logger.info(
            f"[INDEX] ✅ Completed: {documents_indexed} docs | {chunks_indexed} chunks | {duration:.2f}s"
        )

        return {
            "documents_indexed": documents_indexed,
            "chunks_indexed": chunks_indexed,
            "duration_seconds": duration,
        }

    except Exception as e:
        logger.error(f"[INDEX] ❌ Ingestion failed: {e}")
        raise


@shared_task(bind=True)
def reindex_document_task(self, document_id: str):
    """
    Reindex a specific document by deleting existing chunks and re-chunking/re-indexing.

    Implements T092: Reindex endpoint.

    Args:
        document_id: Document ID (UUID string)

    Returns:
        dict: {
            "document_id": str,
            "chunks_created": int,
            "duration_seconds": float,
        }
    """
    import time
    from uuid import UUID

    from .core.vectorize import qdrant_client
    from .core.vectorize import settings as vectorize_settings
    from .database import SessionLocal
    from .models import Chunk, Document
    from .services.elasticsearch import es_client
    from .services.elasticsearch import settings as es_settings

    start_time = time.time()

    try:
        doc_uuid = UUID(document_id)

        with SessionLocal() as db:
            # Get document
            doc = db.query(Document).filter(Document.id == doc_uuid).first()

            if not doc:
                raise ValueError(f"Document not found: {document_id}")

            # Get existing chunks
            existing_chunks = db.query(Chunk).filter(Chunk.documentId == doc_uuid).all()
            chunk_ids = [str(chunk.id) for chunk in existing_chunks]

            # Delete from Qdrant
            if chunk_ids:
                try:
                    qdrant_client.delete(
                        collection_name=vectorize_settings.qdrant_collection_name,
                        points_selector=chunk_ids,
                    )
                except Exception as e:
                    logger.warning(f"[REINDEX] Failed to delete from Qdrant: {e}")

            # Delete from Elasticsearch
            if chunk_ids:
                try:
                    for chunk_id in chunk_ids:
                        try:
                            es_client.delete(
                                index=es_settings.elasticsearch_index,
                                id=chunk_id,
                                ignore=[404],
                            )
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(
                        f"[REINDEX] Failed to delete from Elasticsearch: {e}"
                    )

            # Delete chunks from database
            for chunk in existing_chunks:
                db.delete(chunk)
            db.commit()

            # Re-chunk and re-index with document metadata
            doc_metadata = doc.metadata_ or {}
            chunk_result = chunk_and_index_document(
                str(doc.id),
                doc.title,
                doc.content,
                metadata=doc_metadata,  # Pass document metadata
            )

            # Update document as indexed
            if doc.metadata_ is None:
                doc.metadata_ = {}
            doc.metadata_["is_indexed"] = True
            db.commit()

        duration = time.time() - start_time
        chunks_created = chunk_result.get("chunks_created", 0)

        logger.info(
            f"[REINDEX] ✅ doc_id={document_id[:8]}... | {len(chunk_ids)} deleted → {chunks_created} created | {duration:.2f}s"
        )

        return {
            "document_id": document_id,
            "chunks_created": chunks_created,
            "duration_seconds": duration,
        }

    except Exception as e:
        logger.error(f"[REINDEX] ❌ Failed: {e}")
        raise