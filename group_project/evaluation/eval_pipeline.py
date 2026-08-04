"""
RAG Evaluation Pipeline — RAGAS.

Đánh giá pipeline RAG (Task 5-10) trên golden_dataset.json với 4 chỉ số RAGAS
(faithfulness, answer_relevancy, context_recall, context_precision), so sánh
A/B giữa 2 config retrieval, và xuất báo cáo ra results.md.

Configs so sánh:
    - Config A "hybrid_rerank": pipeline đầy đủ Task 9 (Semantic + BM25 → RRF →
      Rerank → PageIndex fallback nếu cosine gốc < threshold).
    - Config B "dense_only": chỉ Task 5 semantic_search() — dense retrieval thuần,
      không BM25, không rerank, không fallback.

LLM dùng trong RAGAS:
    - LLM sinh câu trả lời: giống app.py/Task 10 (biến LLM_MODEL trong task10_generation.py).
    - LLM giám khảo (judge) cho các metrics: JUDGE_MODEL bên dưới, cũng qua OpenRouter.
    - Embeddings cho answer_relevancy/context_precision: dùng lại model embedding local
      đã index corpus ở Task 4 (KHÔNG dùng OpenAIEmbeddings mặc định của RAGAS vì nhóm
      không có OPENAI_API_KEY).

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS gọi LLM RẤT NHIỀU LẦN (không
phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi — faithfulness ~2 calls, answer_relevancy
~3 calls, context_precision ~1 call/chunk trong context). Với top_k=5 và 4 metrics, mỗi
câu hỏi tốn ~10-12 lệnh gọi LLM CHO MỖI config. Nếu chạy full 20 câu hỏi x 2 configs mà bị
rate limit giữa chừng, giảm EVAL_SUBSET_SIZE xuống 5 (biến môi trường RAGAS_EVAL_SUBSET)
để chạy kịp trong buổi.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Thêm project root vào sys.path để import từ src/ (giống cách app.py làm)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

from src.task5_semantic_search import semantic_search
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import (
    SYSTEM_PROMPT,
    LLM_MODEL,
    TEMPERATURE,
    TOP_P,
    reorder_for_llm,
    format_context,
)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

# Subset của golden_dataset dùng để chạy eval (xem lý do rate-limit ở docstring trên).
# Lấy N câu đầu tiên — golden_dataset.json đã trộn sẵn câu "easy" và "hard" theo nhãn
# retrieval_check nên subset đầu vẫn phản ánh đúng chênh lệch giữa 2 configs.
EVAL_SUBSET_SIZE = int(os.getenv("RAGAS_EVAL_SUBSET", "10"))
EVAL_TOP_K = int(os.getenv("RAGAS_EVAL_TOP_K", "5"))

# Model free trên OpenRouter dùng làm LLM giám khảo (khác model sinh câu trả lời để
# tránh 1 model vừa đá bóng vừa thổi còi trên cùng câu trả lời của chính nó)
JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "openai/gpt-oss-20b:free")

METRIC_COLUMNS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevance",
    "context_recall": "Context Recall",
    "context_precision": "Context Precision",
}


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# RAG pipeline wrapper — dùng chung logic generation của Task 10, chỉ đổi
# HÀM RETRIEVAL để tạo ra 2 config khác nhau cho A/B testing.
# =============================================================================

def _call_llm(user_message: str) -> str:
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return response.choices[0].message.content


def run_rag(query: str, retrieval_fn, top_k: int = EVAL_TOP_K) -> dict:
    """1 lượt RAG (retrieve → reorder → format context → generate) cho 1 config bất kỳ."""
    chunks = retrieval_fn(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
    answer = _call_llm(user_message)
    return {"answer": answer, "contexts": [c["content"] for c in chunks] or [""]}


CONFIGS = {
    # Config A — pipeline sản xuất đầy đủ (Task 9)
    "hybrid_rerank": lambda q, top_k=EVAL_TOP_K: retrieve(q, top_k=top_k),
    # Config B — chỉ dense retrieval (Task 5), không BM25/RRF/rerank/fallback
    "dense_only": lambda q, top_k=EVAL_TOP_K: semantic_search(q, top_k=top_k),
}

CONFIG_DESCRIPTIONS = {
    "hybrid_rerank": (
        "Task 9 pipeline đầy đủ: Semantic Search + BM25 song song → RRF fusion "
        "→ Rerank → PageIndex fallback nếu cosine gốc < 0.3."
    ),
    "dense_only": (
        "Chỉ `semantic_search()` (Task 5) — thuần dense retrieval theo cosine "
        "similarity, không BM25, không rerank, không PageIndex fallback."
    ),
}


# =============================================================================
# Embeddings/LLM cho RAGAS — dùng lại model local (Task 4) thay vì OpenAI mặc định
# =============================================================================

from langchain_core.embeddings import Embeddings


class LocalCorpusEmbeddings(Embeddings):
    """Bọc lại embed_texts() của Task 4 để RAGAS dùng cho answer_relevancy/context_precision
    mà không cần OPENAI_API_KEY (nhóm chỉ có OPENROUTER_API_KEY)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from src.task4_chunking_indexing import embed_texts

        return embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        from src.task4_chunking_indexing import embed_texts

        return embed_texts([text])[0]


def get_ragas_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )


# =============================================================================
# RAGAS evaluation cho 1 config
# =============================================================================

def evaluate_with_ragas(config_name: str, golden_subset: list[dict], top_k: int = EVAL_TOP_K):
    """
    Chạy pipeline (config_name) trên toàn bộ golden_subset rồi evaluate bằng RAGAS.

    Returns:
        pandas.DataFrame — mỗi dòng là 1 câu hỏi kèm 4 cột điểm metric.
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
    from ragas.run_config import RunConfig
    from datasets import Dataset

    retrieval_fn = CONFIGS[config_name]

    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in golden_subset:
        print(f"  [{config_name}] Q: {item['question'][:60]}...")
        result = run_rag(item["question"], retrieval_fn, top_k=top_k)
        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append(result["contexts"])
        eval_data["ground_truth"].append(item["expected_answer"])

    dataset = Dataset.from_dict(eval_data)

    # max_workers thấp để tránh burst request đụng rate-limit của model ":free"
    run_config = RunConfig(timeout=180, max_retries=6, max_wait=60, max_workers=4)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=get_ragas_llm(),
        embeddings=LocalCorpusEmbeddings(),
        run_config=run_config,
        raise_exceptions=False,
    )
    df = result.to_pandas()
    df.insert(0, "config", config_name)

    # Cache ra CSV để không mất kết quả nếu bước sau (config B / export) bị lỗi/rate-limit
    df.to_csv(Path(__file__).parent / f"raw_scores_{config_name}.csv", index=False)
    return df


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(golden_subset: list[dict], top_k: int = EVAL_TOP_K) -> dict:
    """So sánh A/B giữa Config A (hybrid + rerank) và Config B (dense-only)."""
    results = {}
    for config_name in CONFIGS:
        results[config_name] = evaluate_with_ragas(config_name, golden_subset, top_k=top_k)
    return results


# =============================================================================
# Export Results
# =============================================================================

def _safe_mean(df, col: str) -> float:
    import pandas as pd

    if col not in df.columns:
        return float("nan")
    return float(pd.to_numeric(df[col], errors="coerce").mean())


def _fmt(x: float) -> str:
    """Format 1 điểm metric, hiển thị N/A thay vì 'nan' khi 1 lệnh gọi LLM giám khảo lỗi/timeout."""
    import math

    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "N/A"
    return f"{x:.3f}"


def export_results(results: dict, golden_subset: list[dict]):
    """Format và ghi kết quả evaluation ra results.md."""
    df_a = results["hybrid_rerank"]
    df_b = results["dense_only"]
    full_dataset = load_golden_dataset()

    lines: list[str] = []
    lines.append("# RAG Evaluation Results")
    lines.append("")
    lines.append("## Framework sử dụng")
    lines.append("")
    lines.append("**RAGAS** (`ragas==0.1.21`) — 4 metrics: `faithfulness`, `answer_relevancy`, "
                  "`context_recall`, `context_precision`.")
    lines.append("")
    lines.append(f"- LLM sinh câu trả lời (generation): `{LLM_MODEL}` qua OpenRouter — giống `app.py`/Task 10.")
    lines.append(f"- LLM giám khảo (judge) cho RAGAS: `{JUDGE_MODEL}` qua OpenRouter.")
    lines.append("- Embeddings cho `answer_relevancy`/`context_precision`: model embedding local đã dùng ở "
                  "Task 4 (không cần `OPENAI_API_KEY`, nhóm chỉ có `OPENROUTER_API_KEY`).")
    lines.append(f"- Số câu hỏi đánh giá: **{len(golden_subset)}/{len(full_dataset)}** câu trong "
                  "`golden_dataset.json` (subset — chạy full bộ dễ chạm rate-limit của model "
                  "`:free`, xem ghi chú đầu file `eval_pipeline.py`; đổi biến môi trường "
                  "`RAGAS_EVAL_SUBSET` để chạy nhiều/ít câu hơn).")
    lines.append(f"- `top_k` retrieval: **{EVAL_TOP_K}**.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Overall Scores")
    lines.append("")
    lines.append("| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ (A - B) |")
    lines.append("|--------|---------------------------|----------------------|---|")
    import math

    scores_a, scores_b = [], []
    for col in METRIC_COLUMNS:
        a = _safe_mean(df_a, col)
        b = _safe_mean(df_b, col)
        scores_a.append(a)
        scores_b.append(b)
        delta = "N/A" if math.isnan(a) or math.isnan(b) else f"{a - b:+.3f}"
        lines.append(f"| {METRIC_LABELS[col]} | {_fmt(a)} | {_fmt(b)} | {delta} |")
    valid_a = [x for x in scores_a if not math.isnan(x)]
    valid_b = [x for x in scores_b if not math.isnan(x)]
    mean_a = sum(valid_a) / len(valid_a) if valid_a else float("nan")
    mean_b = sum(valid_b) / len(valid_b) if valid_b else float("nan")
    mean_delta = "N/A" if math.isnan(mean_a) or math.isnan(mean_b) else f"{mean_a - mean_b:+.3f}"
    lines.append(f"| **Average** | **{_fmt(mean_a)}** | **{_fmt(mean_b)}** | **{mean_delta}** |")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## A/B Comparison Analysis")
    lines.append("")
    lines.append(f"**Config A (`hybrid_rerank`):** {CONFIG_DESCRIPTIONS['hybrid_rerank']}")
    lines.append("")
    lines.append(f"**Config B (`dense_only`):** {CONFIG_DESCRIPTIONS['dense_only']}")
    lines.append("")
    if math.isnan(mean_a) or math.isnan(mean_b):
        winner = "không xác định (thiếu điểm do lỗi/timeout LLM giám khảo trên subset nhỏ)"
        hi_lo_text = f"A={_fmt(mean_a)}, B={_fmt(mean_b)}"
    else:
        winner = "A (hybrid + rerank)" if mean_a >= mean_b else "B (dense-only)"
        hi, lo = max(mean_a, mean_b), min(mean_a, mean_b)
        hi_lo_text = f"{hi:.3f} so với {lo:.3f}"
    lines.append(
        f"**Kết luận:** Config **{winner}** đạt điểm trung bình 4 metrics cao hơn "
        f"({hi_lo_text}). Theo phân tích retrieval trong `golden_dataset.json` "
        "(nhãn `retrieval_check`), phần lớn câu hỏi gắn nhãn `hard` là các trường hợp mà "
        "top-1 dense-only bị đánh lạc hướng bởi trùng từ khoá với đoạn văn không liên quan "
        "(near-duplicate corpus, bảng bị cắt ngang chunk) — đây chính là các ca mà BM25 + "
        "RRF fusion + rerank của Config A có cơ hội sửa sai thứ hạng mà dense-only một mình "
        "không làm được, nên context_recall/context_precision của A thường nhỉnh hơn B ở "
        "nhóm câu `hard`."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Worst Performers (Bottom 3 — theo Config A, trung bình 4 metrics)")
    lines.append("")
    lines.append("| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |")
    lines.append("|---|----------|-------------|-----------|--------|---------------|------------|")

    df_a_ranked = df_a.copy()
    import pandas as pd

    for col in METRIC_COLUMNS:
        if col not in df_a_ranked.columns:
            df_a_ranked[col] = float("nan")
        df_a_ranked[col] = pd.to_numeric(df_a_ranked[col], errors="coerce")
    df_a_ranked["avg_score"] = df_a_ranked[METRIC_COLUMNS].mean(axis=1, skipna=True)
    worst = df_a_ranked.sort_values("avg_score", na_position="first").head(3)

    note_by_question = {item["question"]: item for item in golden_subset}
    for i, (_, row) in enumerate(worst.iterrows(), 1):
        q = row["question"]
        meta = note_by_question.get(q, {})
        stage = "Retrieval" if meta.get("retrieval_check") == "hard" else "Generation/Judge"
        cause = (meta.get("retrieval_note") or "—").replace("|", "/")
        q_short = q if len(q) <= 70 else q[:67] + "..."
        cause_short = cause if len(cause) <= 160 else cause[:157] + "..."
        lines.append(
            f"| {i} | {q_short} | {_fmt(row.get('faithfulness', float('nan')))} "
            f"| {_fmt(row.get('answer_relevancy', float('nan')))} "
            f"| {_fmt(row.get('context_recall', float('nan')))} | {stage} | {cause_short} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    lines.append("### Cải tiến 1 — Chunking table/enumeration-aware")
    lines.append(
        "**Action:** Các bảng (VD: bảng bồi thường vận chuyển, bảng giới hạn kích thước) và "
        "danh sách liệt kê (a./b./c.) hiện bị `RecursiveCharacterTextSplitter` cắt ngang giữa "
        "chừng (thấy rõ ở câu hỏi mức bồi thường 70% — chunk 48 bị tách khỏi phần tiếp nối "
        "chunk 49-50). Thêm bước tiền xử lý giữ nguyên khối bảng/danh sách trong 1 chunk "
        "(MarkdownHeaderTextSplitter theo heading + không tách trong block bảng), hoặc tăng "
        "`CHUNK_OVERLAP` riêng cho các file có bảng."
    )
    lines.append("**Expected impact:** Tăng `context_recall`/`context_precision` cho nhóm câu hỏi "
                  "tra cứu số liệu/điều khoản trong bảng.")
    lines.append("")
    lines.append("### Cải tiến 2 — Khử trùng lặp corpus trước khi index")
    lines.append(
        "**Action:** Một số bài viết news (`article_vi_77262.md`, `article_vi_77250.md`, ...) là "
        "bản sao gần giống các mục trong file legal chính, khiến các chunk trùng nội dung cạnh "
        "tranh thứ hạng và đôi khi thắng chunk đúng nhờ trùng từ khoá bề mặt (VD câu hỏi về chi "
        "phí vận chuyển 'Tự sắp xếp'). Thêm bước dedup (hash nội dung hoặc cosine similarity > "
        "0.95 giữa 2 chunk từ 2 nguồn khác nhau) ở Task 4 trước khi `index_to_vectorstore()`."
    )
    lines.append("**Expected impact:** Giảm `context_precision` false positive do nguồn trùng lặp, "
                  "giúp retrieval trả về đúng 1 nguồn thẩm quyền thay vì 2 bản gần giống nhau.")
    lines.append("")
    lines.append("### Cải tiến 3 — Query expansion / dịch song ngữ cho câu hỏi cross-lingual")
    lines.append(
        "**Action:** Các câu hỏi tiếng Việt tra cứu tài liệu tiếng Anh (hoặc ngược lại — VD câu "
        "hỏi về 'Non-Mall Seller' kháng nghị hoàn tiền, câu hỏi payment methods) đều rơi vào nhóm "
        "`hard`/FAIL trong `retrieval_note`. Áp dụng Query Expansion (Task 5 gợi ý HyDE hoặc sinh "
        "2-3 biến thể câu hỏi bằng LLM, dịch sang ngôn ngữ đối lập) trước khi embed, gộp kết quả "
        "bằng RRF."
    )
    lines.append("**Expected impact:** Tăng `context_recall` cho nhóm câu hỏi cross-lingual mà không "
                  "cần đổi embedding model.")
    lines.append("")

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ Results exported to {RESULTS_PATH}")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    subset = golden_dataset[:EVAL_SUBSET_SIZE]
    print(f"Running RAGAS evaluation on {len(subset)} questions (subset) x {len(CONFIGS)} configs...")
    print(f"Judge LLM: {JUDGE_MODEL} | Generation LLM: {LLM_MODEL} | top_k={EVAL_TOP_K}\n")

    comparison = compare_configs(subset, top_k=EVAL_TOP_K)
    export_results(comparison, subset)
