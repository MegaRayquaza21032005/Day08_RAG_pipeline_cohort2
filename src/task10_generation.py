"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
from dotenv import load_dotenv

load_dotenv()

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# Gemini 2.5 Flash: nhanh, chi phí hợp lý cho RAG, vẫn đủ tốt cho tiếng Việt.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    f"models/{GEMINI_MODEL}:generateContent"
)


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    # Pattern: [1, 3, 5, 4, 2] theo thứ tự 1-based để giữ chunk tốt nhất ở đầu,
    # chunk tốt thứ hai ở cuối, còn phần kém hơn nằm giữa prompt.
    important_front = chunks[::2]
    important_back = chunks[1::2][::-1]
    return important_front + important_back


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata") or {}
        source = (
            metadata.get("source")
            or metadata.get("filename")
            or metadata.get("title")
            or f"Source {i}"
        )
        doc_type = metadata.get("doc_type") or metadata.get("type") or "unknown"
        chunk_index = metadata.get("chunk_index", i - 1)
        score = chunk.get("score", 0.0)
        citation = f"{source}#chunk-{chunk_index}"

        context_parts.append(
            f"[Document {i} | Citation: {citation} | Source: {source} | "
            f"Type: {doc_type} | Score: {score:.3f}]\n"
            f"{chunk.get('content', '').strip()}\n"
        )

    return "\n---\n".join(context_parts)


def _get_gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _build_user_message(query: str, context: str) -> str:
    return f"""Context:
{context}

---

Question: {query}

Return the answer in Vietnamese. Use the Citation labels from the context,
for example [Luat_phong_chong_ma_tuy_2021.md#chunk-3]."""


def _call_gemini(system_prompt: str, user_message: str) -> str:
    api_key = _get_gemini_api_key()
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in .env.")

    import requests

    response = requests.post(
        GEMINI_API_URL,
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{system_prompt}\n\n{user_message}"}],
                }
            ],
            "generationConfig": {
                "temperature": TEMPERATURE,
                "topP": TOP_P,
            },
        },
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()

    parts = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    )
    answer = "\n".join(part.get("text", "") for part in parts).strip()
    if not answer:
        raise ValueError("Gemini returned an empty answer.")
    return answer


def _citation_label(chunk: dict, index: int) -> str:
    metadata = chunk.get("metadata") or {}
    source = metadata.get("source") or metadata.get("filename") or f"Source {index}"
    chunk_index = metadata.get("chunk_index", index - 1)
    return f"{source}#chunk-{chunk_index}"


def _fallback_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    lines = [
        "Tôi không thể gọi Gemini trong môi trường hiện tại, nên dưới đây là phần trả lời trích xuất từ các nguồn đã retrieve:"
    ]
    for i, chunk in enumerate(chunks[:3], 1):
        citation = _citation_label(chunk, i)
        snippet = " ".join(chunk.get("content", "").split())[:450]
        lines.append(f"- {snippet} [{citation}]")

    lines.append(
        "Các ý trên chỉ phản ánh nội dung trong context; thông tin ngoài context chưa thể xác minh."
    )
    return "\n".join(lines)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if not query.strip():
        raise ValueError("query must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    # Step 2: Reorder để giảm lost-in-the-middle
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context với citation labels
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = _build_user_message(query, context)

    # Step 5: Call Gemini. Nếu thiếu key/network, dùng fallback extractive để
    # function vẫn trả đúng contract trong demo offline.
    try:
        answer = _call_gemini(SYSTEM_PROMPT, user_message)
    except Exception:
        answer = _fallback_answer(query, reordered)

    # Step 6: Return
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
        "model": GEMINI_MODEL,
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
