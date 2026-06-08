"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import math
import os
import re
from collections import Counter
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"
BGE_M3_MODEL = "BAAI/bge-m3"


def _get_jina_api_key() -> str | None:
    return os.getenv("JINA_API_KEY") or os.getenv("JINA_AI_API_KEY")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity dùng cho BGE-M3 fallback và MMR."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _with_new_score(candidate: dict, score: float) -> dict:
    item = candidate.copy()
    item["original_score"] = candidate.get("score")
    item["score"] = float(score)
    item["rerank_score"] = float(score)
    return item


@lru_cache(maxsize=1)
def _load_bge_m3_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(BGE_M3_MODEL)


def _lexical_fallback_rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Fallback cuối cùng để module vẫn chạy khi chưa có API/model local."""
    query_terms = Counter(_tokenize(query))
    if not query_terms:
        return candidates[:top_k]

    query_set = set(query_terms)
    scored = []
    for candidate in candidates:
        content_terms = Counter(_tokenize(candidate.get("content", "")))
        overlap = sum(min(query_terms[term], content_terms[term]) for term in query_set)
        coverage = overlap / max(1, len(query_terms))
        original_score = float(candidate.get("score") or 0.0)
        score = coverage + 0.05 * original_score
        scored.append(_with_new_score(candidate, score))

    return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not candidates:
        return []

    jina_api_key = _get_jina_api_key()
    if not jina_api_key:
        return rerank_bge_m3(query, candidates, top_k)

    import requests

    response = requests.post(
        JINA_RERANK_URL,
        headers={
            "Authorization": f"Bearer {jina_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": JINA_RERANK_MODEL,
            "query": query,
            "documents": [c["content"] for c in candidates],
            "top_n": top_k,
        },
        timeout=15,
    )
    response.raise_for_status()
    reranked = response.json().get("results", [])
    return [
        _with_new_score(candidates[r["index"]], r["relevance_score"])
        for r in reranked
        if "index" in r and "relevance_score" in r
    ]


def rerank_bge_m3(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Fallback rerank bằng BGE-M3 embedding similarity.

    BGE-M3 không phải cross-encoder reranker; ở đây dùng cùng model để embed query
    và candidate rồi sắp xếp theo cosine similarity.
    """
    if not candidates:
        return []

    try:
        model = _load_bge_m3_model()
        texts = [query] + [candidate.get("content", "") for candidate in candidates]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = [embedding.tolist() for embedding in embeddings]
    except Exception:
        return _lexical_fallback_rerank(query, candidates, top_k)

    query_embedding = embeddings[0]
    scored = []
    for candidate, candidate_embedding in zip(candidates, embeddings[1:]):
        score = cosine_sim(query_embedding, candidate_embedding)
        scored.append(_with_new_score(candidate, score))

    return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []
    if not 0 <= lambda_param <= 1:
        raise ValueError("lambda_param must be between 0 and 1.")

    selected = []
    remaining = list(range(len(candidates)))
    mmr_scores = {}
    
    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')
    
        for idx in remaining:
            # Relevance to query
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])
    
            # Max similarity to already selected
            max_sim_to_selected = 0
            for sel_idx in selected:
                sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)
    
            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
    
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
    
        if best_idx is None:
            break

        mmr_scores[best_idx] = best_score
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    return [_with_new_score(candidates[i], mmr_scores[i]) for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores = {}  # content -> score
    content_map = {}  # content -> full dict
    
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            content_map[key] = item
    
    # Sort by RRF score
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)
    
    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if not query.strip():
        raise ValueError("query must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")
    if not candidates:
        return []

    if method == "cross_encoder":
        try:
            return rerank_cross_encoder(query, candidates, top_k)
        except Exception:
            return rerank_bge_m3(query, candidates, top_k)
    elif method == "bge_m3":
        return rerank_bge_m3(query, candidates, top_k)
    elif method == "mmr":
        model = _load_bge_m3_model()
        texts = [query] + [candidate.get("content", "") for candidate in candidates]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        query_embedding = embeddings[0].tolist()
        candidates_with_embeddings = []
        for candidate, embedding in zip(candidates, embeddings[1:]):
            item = candidate.copy()
            item["embedding"] = embedding.tolist()
            candidates_with_embeddings.append(item)
        return rerank_mmr(query_embedding, candidates_with_embeddings, top_k)
    elif method == "rrf":
        return rerank_rrf([candidates], top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
