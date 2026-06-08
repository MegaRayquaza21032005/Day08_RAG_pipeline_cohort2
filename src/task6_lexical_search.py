"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import re
import math
from collections import Counter
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
BM25_INDEX = None


class SimpleBM25:
    """Fallback BM25 khi môi trường chưa cài rank-bm25."""

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_count = len(tokenized_corpus)
        self.doc_lengths = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_lengths) / self.doc_count if self.doc_count else 0
        self.term_freqs = [Counter(doc) for doc in tokenized_corpus]
        self.idf = self._calculate_idf()

    def _calculate_idf(self) -> dict[str, float]:
        doc_freq = Counter()
        for doc in self.tokenized_corpus:
            doc_freq.update(set(doc))

        return {
            term: math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for idx, term_freq in enumerate(self.term_freqs):
            doc_len = self.doc_lengths[idx]
            score = 0.0

            for token in query_tokens:
                freq = term_freq.get(token, 0)
                if freq == 0:
                    continue

                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += self.idf.get(token, 0.0) * (freq * (self.k1 + 1)) / denominator

            scores.append(score)
        return scores


def _tokenize(text: str) -> list[str]:
    """Tokenize đơn giản, giữ tốt tiếng Việt hơn split() vì bỏ dấu câu."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Chunk theo ký tự để lexical search có cùng đơn vị gần giống Task 4."""
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


def load_corpus() -> list[dict]:
    """
    Load corpus từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    corpus = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in md_file.parts else "news"

        for chunk_index, chunk in enumerate(_chunk_text(content)):
            corpus.append({
                "content": chunk,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                    "doc_type": doc_type,
                    "chunk_index": chunk_index,
                },
            })
    return corpus


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    try:
        from rank_bm25 import BM25Okapi

        return BM25Okapi(tokenized_corpus)
    except ModuleNotFoundError:
        return SimpleBM25(tokenized_corpus)


def _get_bm25_index():
    """Lazy init để import module nhanh, search lần đầu mới build index."""
    global CORPUS, BM25_INDEX

    if not CORPUS:
        CORPUS = load_corpus()
    if not CORPUS:
        raise ValueError(f"No markdown documents found in {STANDARDIZED_DIR}")

    if BM25_INDEX is None:
        BM25_INDEX = build_bm25_index(CORPUS)
    return BM25_INDEX


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not query.strip():
        raise ValueError("query must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    bm25 = _get_bm25_index()
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    
    top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
    return results
    # raise NotImplementedError("Implement lexical_search")


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
