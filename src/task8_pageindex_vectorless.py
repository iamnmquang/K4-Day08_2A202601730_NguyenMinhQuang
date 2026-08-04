"""
Task 8 — PageIndex Vectorless RAG.
"""
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Where we persist the md_file -> doc_id mapping so pageindex_search()
# doesn't need to re-upload documents on every call.
DOC_MAP_PATH = Path(__file__).parent / "pageindex_doc_map.json"

POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS = 600


def _load_doc_map() -> dict:
    if DOC_MAP_PATH.exists():
        return json.loads(DOC_MAP_PATH.read_text(encoding="utf-8"))
    return {}


def _save_doc_map(doc_map: dict) -> None:
    DOC_MAP_PATH.write_text(json.dumps(doc_map, ensure_ascii=False, indent=2), encoding="utf-8")


def _md_to_pdf(md_path: Path, out_dir: Path) -> Path:
    """
    PageIndex's cloud pipeline is built around PDF (and, per current docs,
    Markdown for some endpoints) — but the documented Python SDK path is
    submit_document(<file path>) against a PDF. To stay safe across SDK
    versions we convert .md -> a simple PDF with fpdf2 rather than relying
    on markdown being accepted everywhere.
    """
    from fpdf import FPDF

    text = md_path.read_text(encoding="utf-8")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    # fpdf2's core fonts are latin-1 only; encode defensively so Vietnamese
    # or other unicode content in the markdown doesn't crash the export.
    for line in text.splitlines() or [""]:
        safe_line = line.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe_line)

    out_path = out_dir / (md_path.stem + ".pdf")
    pdf.output(str(out_path))
    return out_path


def upload_documents() -> dict:
    """
    Upload toàn bộ markdown documents lên PageIndex.
    Trả về dict {md_filename: doc_id} và lưu lại vào DOC_MAP_PATH để
    pageindex_search() dùng lại, tránh upload trùng lặp.
    """
    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    doc_map = _load_doc_map()

    tmp_pdf_dir = Path(__file__).parent / "_pageindex_tmp_pdfs"
    tmp_pdf_dir.mkdir(exist_ok=True)

    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        print(f"  (không tìm thấy .md nào trong {STANDARDIZED_DIR})")
        return doc_map

    for md_file in md_files:
        key = str(md_file.relative_to(STANDARDIZED_DIR))
        if key in doc_map:
            print(f"  ↷ Skip (đã upload): {md_file.name} -> {doc_map[key]}")
            continue

        pdf_path = _md_to_pdf(md_file, tmp_pdf_dir)

        resp = client.submit_document(str(pdf_path))
        # In toàn bộ response thô trước khi lấy field ra — SDK/API có thể
        # đổi tên field (doc_id vs id) giữa các phiên bản.
        print(f"  submit_document raw response: {json.dumps(resp, ensure_ascii=False)}")

        doc_id = resp.get("doc_id") or resp.get("id")
        if not doc_id:
            print(f"  ✗ Không lấy được doc_id cho {md_file.name}, bỏ qua.")
            continue

        # Poll cho tới khi tree được build xong trước khi coi là "uploaded".
        deadline = time.time() + POLL_TIMEOUT_SECONDS
        while time.time() < deadline:
            try:
                if client.is_retrieval_ready(doc_id):
                    break
            except Exception as e:
                print(f"    (poll status lỗi, thử lại: {e})")
            time.sleep(POLL_INTERVAL_SECONDS)

        doc_map[key] = doc_id
        print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")

    _save_doc_map(doc_map)
    return doc_map


def _flatten_relevant_contents(node: dict) -> list[dict]:
    """
    Chuẩn hoá 'relevant_contents' về 1 list phẳng các dict content.

    Theo docs.pageindex.ai hiện tại (2026-06), field này là:
        "relevant_contents": [{"page_index": 10, "relevant_content": "..."}]
    tức list[dict], KHÔNG phải list[list[{...}]] như comment cũ trong file
    này giả định. Nhưng vì /retrieval đã deprecated và response thực tế có
    thể khác theo version tài khoản, ta xử lý cả 2 dạng (phẳng và lồng)
    để không crash nếu schema thay đổi.
    """
    raw = node.get("relevant_contents", [])
    flat = []
    for item in raw:
        if isinstance(item, dict):
            flat.append(item)
        elif isinstance(item, list):
            # dạng lồng list[list[{...}]] như trong ví dụ cũ
            for sub in item:
                if isinstance(sub, dict):
                    flat.append(sub)
    return flat


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.
    """
    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    doc_map = _load_doc_map()
    if not doc_map:
        print("  ⚠ Chưa có document nào được upload. Gọi upload_documents() trước.")
        return []

    results: list[dict] = []

    for md_name, doc_id in doc_map.items():
        try:
            submit_resp = client.submit_retrieval_query(doc_id=doc_id, query=query, thinking=False)
        except Exception as e:
            print(f"  ✗ submit_retrieval_query lỗi cho {md_name}: {e}")
            continue

        retrieval_id = submit_resp.get("retrieval_id") or submit_resp.get("id")
        if not retrieval_id:
            print(f"  ✗ Không lấy được retrieval_id cho {md_name}: {json.dumps(submit_resp, ensure_ascii=False)}")
            continue

        # Poll cho đến khi status == "completed"
        retrieval = {}
        deadline = time.time() + POLL_TIMEOUT_SECONDS
        while time.time() < deadline:
            retrieval = client.get_retrieval_result(retrieval_id)
            if retrieval.get("status") == "completed":
                break
            time.sleep(POLL_INTERVAL_SECONDS)

        # QUAN TRỌNG: in response thật ra trước khi parse, vì /retrieval đã
        # deprecated (vẫn chạy nhưng kèm field "deprecation" cảnh báo) — không
        # đoán schema dựa trên ví dụ code cũ.
        print(f"  raw retrieval response ({md_name}): {json.dumps(retrieval, ensure_ascii=False)}")

        if "deprecation" in retrieval:
            print(f"  ⚠ PageIndex deprecation notice: {retrieval['deprecation']}")

        nodes = retrieval.get("retrieved_nodes", [])
        for rank, node in enumerate(nodes[:top_k]):
            node_title = node.get("title") or node.get("section_title") or ""
            for item in _flatten_relevant_contents(node):
                content = item.get("relevant_content", "")
                if not content:
                    continue
                section_title = item.get("section_title") or node_title
                results.append({
                    "content": content,
                    # PageIndex không trả score trực tiếp — tự gán theo rank
                    # (node xuất hiện sớm hơn = liên quan hơn).
                    "score": round(1.0 / (rank + 1), 3),
                    "metadata": {
                        "section": section_title,
                        "doc": md_name,
                        "node_id": node.get("node_id"),
                        "page_index": item.get("page_index"),
                    },
                    "source": "pageindex",
                })

    # Gộp kết quả từ nhiều document, giữ điểm cao nhất lên trước.
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()
        print("\nTest query:")
        results = pageindex_search("danh sách sản phẩm cấm đăng bán", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")