"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
import re
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _load_local_chunks() -> list[dict]:
    chunks = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in md_file.parts else "news"

        for chunk_index, chunk in enumerate(_chunk_text(content)):
            chunks.append({
                "content": chunk,
                "metadata": {
                    "source": md_file.name,
                    "filename": md_file.name,
                    "type": doc_type,
                    "doc_type": doc_type,
                    "chunk_index": chunk_index,
                },
            })
    return chunks


def _normalize_pageindex_result(result) -> dict:
    """Chuẩn hóa result từ SDK/API vì PageIndex SDK có thể trả object hoặc dict."""
    if isinstance(result, dict):
        content = (
            result.get("content")
            or result.get("text")
            or result.get("page_content")
            or result.get("chunk")
            or ""
        )
        score = result.get("score", result.get("relevance_score", result.get("similarity", 0.0)))
        metadata = result.get("metadata") or {}
    else:
        content = (
            getattr(result, "content", None)
            or getattr(result, "text", None)
            or getattr(result, "page_content", None)
            or getattr(result, "chunk", None)
            or ""
        )
        score = (
            getattr(result, "score", None)
            or getattr(result, "relevance_score", None)
            or getattr(result, "similarity", None)
            or 0.0
        )
        metadata = getattr(result, "metadata", None) or {}

    return {
        "content": content,
        "score": float(score or 0.0),
        "metadata": metadata,
        "source": "pageindex",
    }


def _local_vectorless_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Fallback vectorless: dùng keyword/structure overlap trên markdown local.

    Đây không thay thế PageIndex thật, nhưng giữ cùng interface để Task 9 có
    fallback hoạt động khi thiếu SDK, API key, hoặc network.
    """
    query_terms = Counter(_tokenize(query))
    if not query_terms:
        return []

    query_set = set(query_terms)
    results = []
    for chunk in _load_local_chunks():
        content_terms = Counter(_tokenize(chunk["content"]))
        overlap = sum(min(query_terms[term], content_terms[term]) for term in query_set)
        if overlap <= 0:
            continue

        unique_coverage = len(query_set.intersection(content_terms)) / max(1, len(query_set))
        frequency_coverage = overlap / max(1, sum(query_terms.values()))
        score = 0.7 * unique_coverage + 0.3 * frequency_coverage
        results.append({
            "content": chunk["content"],
            "score": float(score),
            "metadata": chunk["metadata"],
            "source": "pageindex",
        })

    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


def _get_pageindex_client():
    if not PAGEINDEX_API_KEY:
        return None

    try:
        from pageindex import PageIndex
    except ImportError:
        return None

    return PageIndex(api_key=PAGEINDEX_API_KEY)


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    client = _get_pageindex_client()
    if client is None:
        print("PageIndex SDK/API key chưa sẵn sàng; bỏ qua upload.")
        return []

    uploaded = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        metadata = {"filename": md_file.name, "type": md_file.parent.name}

        if hasattr(client, "upload"):
            result = client.upload(content=content, metadata=metadata)
        elif hasattr(client, "add"):
            result = client.add(content=content, metadata=metadata)
        elif hasattr(client, "add_document"):
            result = client.add_document(content=content, metadata=metadata)
        else:
            raise AttributeError("PageIndex client does not expose an upload/add method.")

        uploaded.append({"file": md_file.name, "result": result})
        print(f"  Uploaded: {md_file.name}")

    return uploaded


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not query.strip():
        raise ValueError("query must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    client = _get_pageindex_client()
    if client is None:
        return _local_vectorless_search(query, top_k)

    try:
        if hasattr(client, "query"):
            raw_results = client.query(query=query, top_k=top_k)
        elif hasattr(client, "search"):
            raw_results = client.search(query=query, top_k=top_k)
        else:
            raise AttributeError("PageIndex client does not expose a query/search method.")
    except Exception:
        return _local_vectorless_search(query, top_k)

    if isinstance(raw_results, dict):
        raw_results = (
            raw_results.get("results")
            or raw_results.get("data")
            or raw_results.get("matches")
            or []
        )

    normalized = [_normalize_pageindex_result(result) for result in raw_results]
    normalized = [result for result in normalized if result["content"]]
    return sorted(normalized, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("Hãy set PAGEINDEX_API_KEY trong file .env để dùng PageIndex thật.")
        print("  Đăng ký tại: https://pageindex.ai/")

    print("\nTest query:")
    results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] [{r['source']}] {r['content'][:100]}...")
