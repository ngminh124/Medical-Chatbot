"""
Hybrid Search Module
Combines vector search (Qdrant) with BM25 search using RRF fusion.
"""
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional

from loguru import logger

from ..configs.setup import get_backend_settings
from ..services.elastic_search import get_elasticsearch_client
from ..services.embedding import get_embedding_service
from .vectorize import search_vectors_for_hybrid

settings = get_backend_settings()


def reciprocal_rank_fusion(
    results_lists: List[List[Dict]],
    k: int = 60,
) -> List[Dict]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion (RRF).
    
    Formula: RRF(d) = Σ 1/(k + rank(d)) for each result list
    
    Args:
        results_lists: List of result lists, each sorted by relevance
        k: RRF constant (default 60, as per original paper)
        
    Returns:
        Combined and re-ranked results
    """
    fused_scores: Dict[str, float] = {}
    result_data: Dict[str, Dict] = {}
    
    for results in results_lists:
        for rank, result in enumerate(results, start=1):
            doc_id = result.get("chunk_id") or result.get("id") or str(rank)
            
            # RRF score calculation
            rrf_score = 1.0 / (k + rank)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + rrf_score
            
            # Keep the result data (use first occurrence)
            if doc_id not in result_data:
                result_data[doc_id] = result
    
    # Sort by fused score
    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    
    # Build final results with RRF scores
    final_results = []
    for doc_id in sorted_ids:
        result = result_data[doc_id].copy()
        result["rrf_score"] = fused_scores[doc_id]
        result["search_type"] = "hybrid"
        final_results.append(result)
    
    return final_results


def hybrid_search(
    query: str,
    top_k: int = None,
    vector_k: int = None,
    bm25_k: int = None,
    final_k: int = None,
    collection_name: str = None,
    use_bm25: bool = True,
    use_vector: bool = True,
    doc_type_filter: Optional[str] = None,
    source_filter: Optional[str] = None,
) -> List[Dict]:
    """
    Perform hybrid search combining vector search and BM25.
    
    Args:
        query: Search query string
        top_k: Number of results to return
        collection_name: Qdrant collection name
        use_bm25: Whether to include BM25 results
        use_vector: Whether to include vector search results
        doc_type_filter: Filter by document type
        source_filter: Filter by source
        
    Returns:
        List of search results with RRF fusion scores
    """
    vector_k = int(vector_k or settings.vector_k)
    bm25_k = int(bm25_k or settings.bm25_k)
    final_k = int(final_k or top_k or settings.final_k)

    if final_k <= 0:
        final_k = settings.final_k
    if vector_k <= 0:
        vector_k = settings.vector_k
    if bm25_k <= 0:
        bm25_k = settings.bm25_k

    collection_name = collection_name or settings.default_collection_name

    logger.info(
        f"[RETRIEVAL] vector_k={vector_k}, bm25_k={bm25_k}, final_k={final_k}"
    )
    
    results_lists = []

    def _run_vector() -> List[Dict]:
        if not use_vector:
            return []
        try:
            embedding_service = get_embedding_service()
            query_vector = embedding_service.embed_query(query)
            if not query_vector:
                return []
            return search_vectors_for_hybrid(
                query_vector=query_vector,
                top_k=vector_k,
                collection_name=collection_name,
                doc_type_filter=doc_type_filter,
                source_filter=source_filter,
            )
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    def _run_bm25() -> List[Dict]:
        if not use_bm25:
            return []
        try:
            return _bm25_search(
                query=query,
                top_k=bm25_k,
                doc_type_filter=doc_type_filter,
                source_filter=source_filter,
            )
        except Exception as e:
            logger.warning(f"BM25 search error (may not be configured): {e}")
            return []

    with ThreadPoolExecutor(max_workers=2) as executor:
        vector_future = executor.submit(_run_vector)
        bm25_future = executor.submit(_run_bm25)
        vector_results = vector_future.result()
        bm25_results = bm25_future.result()

    if vector_results:
        results_lists.append(vector_results)
        logger.debug(f"Vector search returned {len(vector_results)} results")
    if bm25_results:
        results_lists.append(bm25_results)
        logger.debug(f"BM25 search returned {len(bm25_results)} results")
    
    # Fusion
    if not results_lists:
        logger.warning("No search results from any source")
        return []
    
    if len(results_lists) == 1:
        # Single source - just return as-is with limit
        return results_lists[0][:final_k]
    
    # RRF fusion
    fused_results = reciprocal_rank_fusion(results_lists)
    logger.info(f"Hybrid search returned {len(fused_results[:final_k])} results after RRF fusion")
    
    return fused_results[:final_k]


def _bm25_search(
    query: str,
    top_k: int,
    doc_type_filter: Optional[str] = None,
    source_filter: Optional[str] = None,
) -> List[Dict]:
    """
    BM25 search using Elasticsearch.
    
    Args:
        query: Search query
        top_k: Number of results
        doc_type_filter: Optional document type filter
        source_filter: Optional source filter
        
    Returns:
        List of BM25 search results
    """
    try:
        es_client = get_elasticsearch_client()
        results = es_client.search_bm25(
            query=query,
            top_k=top_k,
            doc_type_filter=doc_type_filter,
            source_filter=source_filter,
        )
        return results
    except Exception as e:
        logger.warning(f"BM25 search unavailable (Elasticsearch may not be running): {e}")
        return []


def vector_only_search(
    query: str,
    top_k: int = None,
    collection_name: str = None,
) -> List[Dict]:
    """
    Convenience function for vector-only search.
    
    Args:
        query: Search query string
        top_k: Number of results
        collection_name: Qdrant collection name
        
    Returns:
        List of vector search results
    """
    return hybrid_search(
        query=query,
        top_k=top_k,
        collection_name=collection_name,
        use_bm25=False,
        use_vector=True,
    )
