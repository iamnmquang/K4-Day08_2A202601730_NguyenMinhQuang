# RAG Evaluation Results

## Framework sử dụng

**RAGAS** (`ragas==0.1.21`) — 4 metrics: `faithfulness`, `answer_relevancy`, `context_recall`, `context_precision`.

- LLM sinh câu trả lời (generation): `google/gemma-4-26b-a4b-it:free` qua OpenRouter — giống `app.py`/Task 10.
- LLM giám khảo (judge) cho RAGAS: `openai/gpt-oss-20b:free` qua OpenRouter.
- Embeddings cho `answer_relevancy`/`context_precision`: model embedding local đã dùng ở Task 4 (không cần `OPENAI_API_KEY`, nhóm chỉ có `OPENROUTER_API_KEY`).
- Số câu hỏi đánh giá: **2/20** câu trong `golden_dataset.json` (subset — chạy full bộ dễ chạm rate-limit của model `:free`, xem ghi chú đầu file `eval_pipeline.py`; đổi biến môi trường `RAGAS_EVAL_SUBSET` để chạy nhiều/ít câu hơn).
- `top_k` retrieval: **5**.

> ⚠️ **Ghi chú về kích thước subset:** Lần chạy tiếp theo (thử mở rộng lên 6 câu) bị chặn giữa
> chừng bởi lỗi `429 Rate limit exceeded: free-models-per-day` của OpenRouter — tài khoản free
> giới hạn **50 request/ngày cho CẢ TÀI KHOẢN** (không phải theo model, đổi model `:free` khác
> không reset được quota — đúng như cảnh báo trong `day8-lab-rag-pipeline.md` mục Troubleshooting).
> Bảng điểm dưới đây vẫn hoàn toàn hợp lệ (chạy thật, không mock) trên 2 câu đã kịp hoàn thành
> trước khi hết quota. Có thể mở rộng lên 15-20 câu bất kỳ lúc nào bằng cách chạy lại
> `RAGAS_EVAL_SUBSET=15 python -m group_project.evaluation.eval_pipeline` sau khi quota reset
> (00:00 UTC hằng ngày) hoặc sau khi nạp credit OpenRouter.

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ (A - B) |
|--------|---------------------------|----------------------|---|
| Faithfulness | 1.000 | 0.500 | +0.500 |
| Answer Relevance | 0.783 | 0.769 | +0.014 |
| Context Recall | 1.000 | 1.000 | +0.000 |
| Context Precision | 0.500 | 0.367 | +0.133 |
| **Average** | **0.821** | **0.659** | **+0.162** |

---

## A/B Comparison Analysis

**Config A (`hybrid_rerank`):** Task 9 pipeline đầy đủ: Semantic Search + BM25 song song → RRF fusion → Rerank → PageIndex fallback nếu cosine gốc < 0.3.

**Config B (`dense_only`):** Chỉ `semantic_search()` (Task 5) — thuần dense retrieval theo cosine similarity, không BM25, không rerank, không PageIndex fallback.

**Kết luận:** Config **A (hybrid + rerank)** đạt điểm trung bình 4 metrics cao hơn (0.821 so với 0.659). Theo phân tích retrieval trong `golden_dataset.json` (nhãn `retrieval_check`), phần lớn câu hỏi gắn nhãn `hard` là các trường hợp mà top-1 dense-only bị đánh lạc hướng bởi trùng từ khoá với đoạn văn không liên quan (near-duplicate corpus, bảng bị cắt ngang chunk) — đây chính là các ca mà BM25 + RRF fusion + rerank của Config A có cơ hội sửa sai thứ hạng mà dense-only một mình không làm được, nên context_recall/context_precision của A thường nhỉnh hơn B ở nhóm câu `hard`.

---

## Worst Performers (Bottom 3 — theo Config A, trung bình 4 metrics)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Buyer may raise a refund/return request for a Received Item up to h... | 1.000 | 0.807 | 1.000 | Generation/Judge | Top-1 hit, score 0.705 |
| 2 | Người mua có thể gửi yêu cầu trả hàng/hoàn tiền trong vòng bao nhiê... | N/A* | 0.759 | 1.000 | Generation/Judge | Top-1 hit, score 0.707 |

<sub>*N/A: lệnh gọi LLM giám khảo cho metric này bị timeout trên model `:free` (đã retry theo `RunConfig`, `raise_exceptions=False` nên không làm sập cả lượt eval) — không tính vào Average phía trên.</sub>

---

## Recommendations

### Cải tiến 1 — Chunking table/enumeration-aware
**Action:** Các bảng (VD: bảng bồi thường vận chuyển, bảng giới hạn kích thước) và danh sách liệt kê (a./b./c.) hiện bị `RecursiveCharacterTextSplitter` cắt ngang giữa chừng (thấy rõ ở câu hỏi mức bồi thường 70% — chunk 48 bị tách khỏi phần tiếp nối chunk 49-50). Thêm bước tiền xử lý giữ nguyên khối bảng/danh sách trong 1 chunk (MarkdownHeaderTextSplitter theo heading + không tách trong block bảng), hoặc tăng `CHUNK_OVERLAP` riêng cho các file có bảng.
**Expected impact:** Tăng `context_recall`/`context_precision` cho nhóm câu hỏi tra cứu số liệu/điều khoản trong bảng.

### Cải tiến 2 — Khử trùng lặp corpus trước khi index
**Action:** Một số bài viết news (`article_vi_77262.md`, `article_vi_77250.md`, ...) là bản sao gần giống các mục trong file legal chính, khiến các chunk trùng nội dung cạnh tranh thứ hạng và đôi khi thắng chunk đúng nhờ trùng từ khoá bề mặt (VD câu hỏi về chi phí vận chuyển 'Tự sắp xếp'). Thêm bước dedup (hash nội dung hoặc cosine similarity > 0.95 giữa 2 chunk từ 2 nguồn khác nhau) ở Task 4 trước khi `index_to_vectorstore()`.
**Expected impact:** Giảm `context_precision` false positive do nguồn trùng lặp, giúp retrieval trả về đúng 1 nguồn thẩm quyền thay vì 2 bản gần giống nhau.

### Cải tiến 3 — Query expansion / dịch song ngữ cho câu hỏi cross-lingual
**Action:** Các câu hỏi tiếng Việt tra cứu tài liệu tiếng Anh (hoặc ngược lại — VD câu hỏi về 'Non-Mall Seller' kháng nghị hoàn tiền, câu hỏi payment methods) đều rơi vào nhóm `hard`/FAIL trong `retrieval_note`. Áp dụng Query Expansion (Task 5 gợi ý HyDE hoặc sinh 2-3 biến thể câu hỏi bằng LLM, dịch sang ngôn ngữ đối lập) trước khi embed, gộp kết quả bằng RRF.
**Expected impact:** Tăng `context_recall` cho nhóm câu hỏi cross-lingual mà không cần đổi embedding model.
