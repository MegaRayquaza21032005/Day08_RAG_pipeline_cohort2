"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

from concurrent.futures import ThreadPoolExecutor

try:
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank, rerank_rrf
    from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"


def _semantic_search(query: str, top_k: int) -> list[dict]:
    """Lazy import để thiếu weaviate/sentence-transformers không làm hỏng Task 9."""
    try:
        from .task5_semantic_search import semantic_search
    except ImportError:
        try:
            from task5_semantic_search import semantic_search
        except ImportError:
            return []

    return semantic_search(query, top_k=top_k)


def _safe_search(search_fn, query: str, top_k: int) -> list[dict]:
    """Chạy một retriever và trả list rỗng nếu retriever chưa sẵn sàng."""
    try:
        results = search_fn(query, top_k=top_k)
    except Exception:
        return []

    return results if isinstance(results, list) else []


def _normalize_result(result: dict, source: str) -> dict:
    metadata = result.get("metadata") or {}
    item = {
        "content": result.get("content", ""),
        "score": float(result.get("score") or 0.0),
        "metadata": metadata,
        "source": source,
    }
    if "original_score" in result:
        item["original_score"] = result["original_score"]
    if "rerank_score" in result:
        item["rerank_score"] = result["rerank_score"]
    return item


def _fallback_pageindex(query: str, top_k: int) -> list[dict]:
    results = _safe_search(pageindex_search, query, top_k)
    return [_normalize_result(result, "pageindex") for result in results[:top_k]]


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if not query.strip():
        raise ValueError("query must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    retrieval_k = max(top_k * 2, top_k)

    # Step 1: Chạy semantic + lexical song song. Mỗi nhánh được guard riêng để
    # pipeline vẫn sống khi Weaviate/model/API chưa sẵn sàng.
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(_safe_search, _semantic_search, query, retrieval_k)
        sparse_future = executor.submit(_safe_search, lexical_search, query, retrieval_k)
        dense_results = dense_future.result()
        sparse_results = sparse_future.result()

    # Step 2: Merge bằng RRF nếu có nhiều ranker, còn nếu chỉ một ranker hoạt
    # động thì dùng trực tiếp kết quả đó.
    ranked_lists = [results for results in [dense_results, sparse_results] if results]
    if ranked_lists:
        merged = rerank_rrf(ranked_lists, top_k=retrieval_k)
        merged = [_normalize_result(result, "hybrid") for result in merged]
    else:
        merged = []

    # Step 3: Rerank lại merged candidates.
    if use_reranking and merged:
        reranked = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        final_results = [_normalize_result(result, "hybrid") for result in reranked]
    else:
        final_results = merged[:top_k]

    # Step 4: Nếu hybrid không có kết quả đủ tin cậy, fallback sang PageIndex.
    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        fallback = _fallback_pageindex(query, top_k)
        if fallback:
            return fallback

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
