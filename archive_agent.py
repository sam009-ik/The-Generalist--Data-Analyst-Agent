# ---------- ARCHIVE AGENT (ZIP / TAR / TAR.GZ) ----------
import os, io, tarfile, zipfile, tempfile, mimetypes
from typing import List, Optional
from starlette.datastructures import UploadFile as StarletteUploadFile
import httpx
import asyncio
from html_agent import html_agent
from pdf_agent import pdf_agent 
from csv_tsv_xlsx_agent import csv_tsv_xlsx_agent #Using Powerdrill
from image_agent import image_agent
from io import BytesIO
from helper_html import render_html_file, render_html_url
ARCHIVE_EXTS = (".zip", ".tar", ".tgz", ".tar.gz")
TABULAR_EXTS = (".csv", ".tsv", ".xlsx")
PDF_EXTS     = (".pdf",)
IMAGE_EXTS   = (".png", ".jpg", ".jpeg", ".webp")
HTML_EXTS    = (".html", ".htm")

MAX_UNPACK_BYTES = 200 * 1024 * 1024   # 200 MB cap (uncompressed)
MAX_FILES        = 200                 # entries per archive
MAX_PER_TYPE     = 50                  # cap per modality

def _safe_join(base, *paths):
    final_path = os.path.abspath(os.path.join(base, *paths))
    if not final_path.startswith(os.path.abspath(base) + os.sep):
        raise ValueError(f"Unsafe path traversal: {final_path}")
    return final_path

def _upload_from_bytes(name: str, data: bytes) -> StarletteUploadFile:
    return StarletteUploadFile(filename=name, file=io.BytesIO(data))

def _collect_archive_blobs_from_urls(urls: List[str]) -> List[tuple[str, bytes]]:
    blobs = []
    for u in (urls or []):
        try:
            r = httpx.get(u, timeout=30, follow_redirects=True)
            print(f"[archive_agent] GET {u} -> {r.status_code}, ct={r.headers.get('content-type')}")
            r.raise_for_status()
            blobs.append((u, r.content))
        except Exception as e:
            print(f"[archive_agent] URL error {u}: {e}")
    return blobs

def _coerce_to_text(result) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        # your CSV agent sometimes returns {"answer": "...", ...}
        return result.get("answer", "") or str(result)
    return str(result)

async def archive_agent(
    task: str,
    archive_files: Optional[List[StarletteUploadFile]] = None,
    archive_urls: Optional[List[str]] = None,
) -> dict:
    """
    Unpack archives and route to your existing agents.
    Returns strings you can pass directly to your master agent contexts:
      {"csv": str, "pdf": str, "image": str, "html": str}
    """
    archive_files = archive_files or []
    archive_urls  = archive_urls  or []

    print(f"[archive_agent] START files={len(archive_files)} urls={len(archive_urls)}")

    # 1) Gather archives (uploads + urls)
    archive_blobs: List[tuple[str, bytes]] = []
    for af in archive_files:
        try:
            data = await af.read()
            print(f"[archive_agent] upload bytes: {af.filename} -> {len(data)}")
            if data:
                archive_blobs.append((af.filename, data))
        except Exception as e:
            print(f"[archive_agent] upload read error {getattr(af,'filename','<upload>')}: {e}")

    archive_blobs += _collect_archive_blobs_from_urls(archive_urls)

    if not archive_blobs:
        print("[archive_agent] no archive payloads")
        return {"csv": "", "pdf": "", "image": "", "html": ""}

    # 2) Extract safely, collect inner files by type
    all_csv:   List[StarletteUploadFile] = []
    all_pdf:   List[StarletteUploadFile] = []
    all_image: List[StarletteUploadFile] = []
    all_html:  List[StarletteUploadFile] = []

    total_unpacked = 0
    total_entries  = 0

    with tempfile.TemporaryDirectory(prefix="arch_") as tdir:
        print(f"[archive_agent] tempdir={tdir}")

        for name, blob in archive_blobs:
            arc_lower = (name or "").lower()
            arc_path = _safe_join(tdir, os.path.basename(name) or "archive.bin")
            with open(arc_path, "wb") as f:
                f.write(blob)
            print(f"[archive_agent] wrote archive -> {arc_path} ({len(blob)} bytes)")

            try:
                if arc_lower.endswith(".zip"):
                    with zipfile.ZipFile(arc_path) as zf:
                        infos = zf.infolist()[:MAX_FILES]
                        for zi in infos:
                            total_entries += 1
                            if zi.is_dir():
                                continue
                            if zi.file_size > MAX_UNPACK_BYTES or total_unpacked + zi.file_size > MAX_UNPACK_BYTES:
                                print(f"[archive_agent] skip large entry {zi.filename} size={zi.file_size}")
                                continue
                            out_path = _safe_join(tdir, zi.filename)
                            os.makedirs(os.path.dirname(out_path), exist_ok=True)
                            with zf.open(zi) as src, open(out_path, "wb") as dst:
                                data = src.read()
                                dst.write(data)
                            total_unpacked += len(data)

                            ext  = os.path.splitext(out_path)[1].lower()
                            base = os.path.basename(out_path)
                            if ext in TABULAR_EXTS and len(all_csv) < MAX_PER_TYPE:
                                all_csv.append(_upload_from_bytes(base, data))
                            elif ext in PDF_EXTS and len(all_pdf) < MAX_PER_TYPE:
                                all_pdf.append(_upload_from_bytes(base, data))
                            elif ext in IMAGE_EXTS and len(all_image) < MAX_PER_TYPE:
                                all_image.append(_upload_from_bytes(base, data))
                            elif ext in HTML_EXTS and len(all_html) < MAX_PER_TYPE:
                                all_html.append(_upload_from_bytes(base, data))

                elif arc_lower.endswith((".tar", ".tgz", ".tar.gz")):
                    mode = "r:gz" if arc_lower.endswith((".tgz", ".tar.gz")) else "r:"
                    with tarfile.open(arc_path, mode) as tf:
                        members = [m for m in tf.getmembers() if m.isfile()][:MAX_FILES]
                        for m in members:
                            total_entries += 1
                            if m.size > MAX_UNPACK_BYTES or total_unpacked + m.size > MAX_UNPACK_BYTES:
                                print(f"[archive_agent] skip large entry {m.name} size={m.size}")
                                continue
                            out_path = _safe_join(tdir, m.name)
                            os.makedirs(os.path.dirname(out_path), exist_ok=True)
                            data = tf.extractfile(m).read()
                            with open(out_path, "wb") as dst:
                                dst.write(data)
                            total_unpacked += len(data)

                            ext  = os.path.splitext(out_path)[1].lower()
                            base = os.path.basename(out_path)
                            if ext in TABULAR_EXTS and len(all_csv) < MAX_PER_TYPE:
                                all_csv.append(_upload_from_bytes(base, data))
                            elif ext in PDF_EXTS and len(all_pdf) < MAX_PER_TYPE:
                                all_pdf.append(_upload_from_bytes(base, data))
                            elif ext in IMAGE_EXTS and len(all_image) < MAX_PER_TYPE:
                                all_image.append(_upload_from_bytes(base, data))
                            elif ext in HTML_EXTS and len(all_html) < MAX_PER_TYPE:
                                all_html.append(_upload_from_bytes(base, data))

                else:
                    print(f"[archive_agent] unsupported archive extension: {name}")
            except Exception as e:
                print(f"[archive_agent] extraction error for {name}: {e}")

        print(f"[archive_agent] extracted entries={total_entries}, bytes={total_unpacked}")
        print(f"[archive_agent] collected csv={len(all_csv)} pdf={len(all_pdf)} image={len(all_image)} html={len(all_html)}")

        # 3) Dispatch to existing agents, producing STRINGS ONLY
        csv_text  = ""
        pdf_text  = ""
        image_text = ""
        html_text = ""

        # CSV/TSV/XLSX (Powerdrill path)
        if all_csv:
            try:
                csv_res = await csv_tsv_xlsx_agent(
                    task_description=task,
                    uploaded_files=all_csv,
                    file_urls=[]
                )
                csv_text = _coerce_to_text(csv_res)
                print(f"[archive_agent] CSV agent len={len(csv_text)}")
            except Exception as e:
                print(f"[archive_agent] CSV agent error: {e}")

        # PDFs (Claude path)
        if all_pdf:
            try:
                pdf_res = await pdf_agent(pdf_files=all_pdf, pdf_urls=[], task=task)
                pdf_text = _coerce_to_text(pdf_res)
                print(f"[archive_agent] PDF agent len={len(pdf_text)}")
            except Exception as e:
                print(f"[archive_agent] PDF agent error: {e}")

        # Images (Claude path)
        if all_image:
            try:
                img_res = await image_agent(image_files=all_image, image_urls=[], task=task)
                image_text = _coerce_to_text(img_res)
                print(f"[archive_agent] IMAGE agent len={len(image_text)}")
            except Exception as e:
                print(f"[archive_agent] IMAGE agent error: {e}")

        # HTML (your exact flow)
        if all_html:
            try:
                print("[archive_agent] Processing HTML files (no URLs from archive)")
                rendered_html_file = await render_html_file(all_html)
                print("[archive_agent] Rendered HTML from files length:", len(rendered_html_file or ""))

                rendered_html_urls = ""  # archives won’t include URL links; keep for symmetry
                full_html = (rendered_html_file or "") + (rendered_html_urls or "")

                if full_html.strip():
                    structured_html = await html_agent(full_html, task)
                    html_text = _coerce_to_text(structured_html)
                    print(f"[archive_agent] HTML agent len={len(html_text)}")
            except Exception as e:
                print(f"[archive_agent] HTML agent error: {e}")

    # 4) Return strings ready for your contexts
    return {
        "csv":   csv_text  or "",
        "pdf":   pdf_text  or "",
        "image": image_text or "",
        "html":  html_text or ""
    }


async def main():
    zip_path = "img_html_pdf_files.zip"
    if not os.path.exists(zip_path):
        print(f"[test-archive] Missing archive: {zip_path}")
        return

    # Build UploadFile from local zip
    with open(zip_path, "rb") as f:
        zip_bytes = f.read()
    archive_files = [StarletteUploadFile(filename=os.path.basename(zip_path), file=BytesIO(zip_bytes))]
    archive_urls = []  # you can put a .zip URL here later if you want

    task = (
        'Look at the zipped file: img_html_pdf_files.zip, Get all files from within and answer the following questions.\n'
        '1. From the image file please detail the complete timeline with the Task. Return a json object like:\n'
        '```\n'
        ' {\n'
        '     "Date": "1st July 2025",\n'
        '     "Task": "Task"\n'
        ' }\n'
        '```\n'
        'for all dates and tasks.\n\n'
        '2. From the pdf get the BHS avg weighted (rs/mt) cost for Repair and Maintenance.\n\n'
        "3. From the html count the number of times the word 'Lorem' appears in the text.\n\n"
        'Finally return a single json array answering all three questions.'
    )

    print("[test-archive] Calling archive_agent ...")
    result = await archive_agent(
        task=task,
        archive_files=archive_files,
        archive_urls=archive_urls
    )
    # result is a dict of strings: {"csv": str, "pdf": str, "image": str, "html": str}
    print("\n=== archive_agent RESULT (per modality) ===")
    for k in ("image", "pdf", "html", "csv"):
        v = result.get(k, "")
        preview = (v[:500] + "...") if isinstance(v, str) and len(v) > 500 else v
        print(f"[{k.upper()}]\n{preview}\n")

    # If you want to pass straight into your master agent the same way you do elsewhere:
    # (each context is a single dict with combined string content)
    image_context = [{"source": {"archive_sources": [zip_path]}, "content": result.get("image", "")}] if result.get("image") else []
    pdf_context   = [{"source": {"archive_sources": [zip_path]}, "content": result.get("pdf",   "")}] if result.get("pdf")   else []
    html_context  = [{"source": {"archive_sources": [zip_path]}, "content": result.get("html",  "")}] if result.get("html")  else []
    csv_context   = [{"source": {"archive_sources": [zip_path]}, "content": result.get("csv",   "")}] if result.get("csv")   else []

    # Example: call your existing data_analyst_agent (uncomment if you want to actually run it)
    # llm_code = await data_analyst_agent(
    #     task=task,
    #     previews=None,
    #     html_context=html_context,
    #     pdf_context=pdf_context,
    #     csv_context=csv_context,
    #     image_context=image_context
    # )
    # print("\n=== data_analyst_agent (code or answer) ===")
    # print((llm_code[:1000] + "...") if isinstance(llm_code, str) and len(llm_code) > 1000 else llm_code)

if __name__ == "__main__":
    asyncio.run(main())