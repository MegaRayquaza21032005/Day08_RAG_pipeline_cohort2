"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""
from dotenv import load_dotenv

load_dotenv()

import os

import weaviate
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery

COLLECTION_NAME = "DrugLawDocs"
EMBEDDING_MODEL = "BAAI/bge-m3"


def _get_weaviate_url() -> str:
    """Read Weaviate Cloud URL from .env and normalize it for the client."""
    weaviate_url = os.getenv("WEAVIATE_URL")
    if not weaviate_url:
        raise ValueError("Missing WEAVIATE_URL. Please add it to .env.")

    if not weaviate_url.startswith(("http://", "https://")):
        weaviate_url = f"https://{weaviate_url}"
    return weaviate_url


def _connect_weaviate():
    """Connect to the same Weaviate Cloud instance used in Task 4."""
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    if not weaviate_api_key:
        raise ValueError("Missing WEAVIATE_API_KEY. Please add it to .env.")

    return weaviate.connect_to_weaviate_cloud(
        cluster_url=_get_weaviate_url(),
        auth_credentials=Auth.api_key(weaviate_api_key),
        additional_config=wvc.init.AdditionalConfig(
            timeout=wvc.init.Timeout(init=60, query=60, insert=120)
        ),
        skip_init_checks=True,
    )


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not query.strip():
        raise ValueError("query must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    # Bước 1: Embed query bằng cùng model ở Task 4.
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode(query).tolist()

    # Bước 2: Query vector store bằng cosine distance trong Weaviate.
    client = _connect_weaviate()
    try:
        collection = client.collections.get(COLLECTION_NAME)
        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )

        # Bước 3: Chuẩn hóa output theo format yêu cầu.
        search_results = []
        for obj in results.objects:
            distance = obj.metadata.distance
            score = 1 - distance if distance is not None else None
            search_results.append({
                "content": obj.properties.get("content", ""),
                "score": score,
                "metadata": {
                    "source": obj.properties.get("source"),
                    "doc_type": obj.properties.get("doc_type"),
                    "chunk_index": obj.properties.get("chunk_index"),
                },
            })

        return sorted(
            search_results,
            key=lambda item: item["score"] if item["score"] is not None else float("-inf"),
            reverse=True,
        )
    finally:
        client.close()


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        source = r["metadata"]["source"]
        chunk_index = r["metadata"]["chunk_index"]
        print(f"[{r['score']:.3f}] {source}#{chunk_index}: {r['content'][:100]}...")
