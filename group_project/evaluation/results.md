# RAG Evaluation Results

Generated at: `2026-06-08T20:18:00`

## Framework sử dụng

Framework: **Lightweight heuristic evaluator**. Script chấm đủ 4 metric bắt buộc bằng token overlap để chạy offline, không phụ thuộc LLM judge.

## Dataset

- Golden samples: **3** (NEEDS MORE DATA; yêu cầu >= 15)
- File: `group_project/evaluation/golden_dataset.json`

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (no rerank) | Delta |
|--------|-----------------------------|----------------------|-------|
| Faithfulness | 0.839 | 0.898 | -0.060 |
| Answer Relevance | 0.079 | 0.075 | +0.004 |
| Context Recall | 0.740 | 0.749 | -0.009 |
| Context Precision | 0.147 | 0.147 | +0.001 |
| Average | 0.451 | 0.467 | -0.016 |

## A/B Comparison Analysis

**Config A:** hybrid retrieval + RRF merge + reranking + generation/fallback citation.

**Config B:** hybrid retrieval + RRF merge, tắt reranking, dùng extractive answer.

**Kết luận:** `Config B - hybrid no rerank extractive` đang có average cao hơn trong bộ golden hiện tại. Vì dataset còn nhỏ, kết luận này chỉ nên xem là smoke test; cần bổ sung đủ 15+ samples để đánh giá ổn định hơn.

## Worst Performers (Bottom 3 - Config A)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|--------------|-----------|--------|---------------|------------|
| 1 | Danh mục các chất ma tuý thuộc nhóm I theo quy định pháp luật Việt Nam gồm những chất nào? | 0.840 | 0.077 | 0.594 | generation | Answer/context overlap với ground truth còn thấp. |
| 2 | Luật Phòng chống ma tuý 2021 quy định những hình thức cai nghiện nào? | 0.806 | 0.073 | 0.839 | generation | Retrieved chunks còn nhiễu hoặc chunk quá dài. |
| 3 | Hình phạt cho tội tàng trữ trái phép chất ma tuý theo Điều 249 Bộ luật Hình sự? | 0.870 | 0.088 | 0.786 | generation | Answer/context overlap với ground truth còn thấp. |

## Per-Case Details (Config A)

| # | Question | Source | Faithfulness | Relevance | Recall | Precision | Average |
|---|----------|--------|--------------|-----------|--------|-----------|---------|
| 1 | Hình phạt cho tội tàng trữ trái phép chất ma tuý theo Điều 249 Bộ luật Hình sự? | hybrid | 0.870 | 0.088 | 0.786 | 0.160 | 0.476 |
| 2 | Luật Phòng chống ma tuý 2021 quy định những hình thức cai nghiện nào? | hybrid | 0.806 | 0.073 | 0.839 | 0.124 | 0.461 |
| 3 | Danh mục các chất ma tuý thuộc nhóm I theo quy định pháp luật Việt Nam gồm những chất nào? | hybrid | 0.840 | 0.077 | 0.594 | 0.158 | 0.417 |

## Recommendations

### Cải tiến 1
**Action:** Bổ sung golden dataset lên ít nhất 15 câu, phủ pháp luật, nghị định, và news.

**Expected impact:** Điểm A/B ổn định hơn và đáp ứng rubric nhóm.

### Cải tiến 2
**Action:** Tối ưu chunking theo heading/điều luật thay vì cắt ký tự thuần.

**Expected impact:** Tăng context precision và citation đúng điều/khoản hơn.

### Cải tiến 3
**Action:** Khi có API ổn định, thêm DeepEval/RAGAS LLM judge để đối chiếu heuristic scores.

**Expected impact:** Đánh giá faithfulness/relevance sát chất lượng câu trả lời hơn.
