# ---------- ARCHIVE AGENT (ZIP / TAR / TAR.GZ) ----------
import json
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
from process_sql_parquet_json import process_sql_parquet_json
from sql_parquet_json_agent import sql_parquet_json_agent, execute_llm_python
from process_sql_parquet_json import process_sql_parquet_json
ARCHIVE_EXTS = (".zip", ".tar", ".tgz", ".tar.gz")
TABULAR_EXTS = (".csv", ".tsv", ".xlsx")
PDF_EXTS     = (".pdf",)
IMAGE_EXTS   = (".png", ".jpg", ".jpeg", ".webp")
HTML_EXTS    = (".html", ".htm")
SQL_PARQUET_JSON_EXTS = (".db", ".sqlite", ".sqlite3", ".duckdb", ".parquet", ".json", ".sql")
SESSION_ROOT = os.getenv("SESSION_ROOT", "/data/_session_sql")

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
        return {"csv": "", "pdf": "", "image": "", "html": "", "sql_parquet_json": ""}

    # 2) Extract safely, collect inner files by type
    all_csv:   List[StarletteUploadFile] = []
    all_pdf:   List[StarletteUploadFile] = []
    all_image: List[StarletteUploadFile] = []
    all_html:  List[StarletteUploadFile] = []
    all_sql_parquet_json: List[StarletteUploadFile] = []

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
                            elif ext in SQL_PARQUET_JSON_EXTS and len(all_sql_parquet_json) < MAX_PER_TYPE:
                                all_sql_parquet_json.append(_upload_from_bytes(base, data))

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
                            elif ext in SQL_PARQUET_JSON_EXTS and len(all_sql_parquet_json) < MAX_PER_TYPE:
                                all_sql_parquet_json.append(_upload_from_bytes(base, data))

                else:
                    print(f"[archive_agent] unsupported archive extension: {name}")
            except Exception as e:
                print(f"[archive_agent] extraction error for {name}: {e}")

        print(f"[archive_agent] extracted entries={total_entries}, bytes={total_unpacked}")
        print(f"[archive_agent] collected csv={len(all_csv)} pdf={len(all_pdf)} image={len(all_image)} html={len(all_html)} sql_parquet_json={len(all_sql_parquet_json)}")

        # 3) Dispatch to existing agents, producing STRINGS ONLY
        csv_text  = ""
        pdf_text  = ""
        image_text = ""
        html_text = ""
        sql_parquet_json_text = ""

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
        if all_sql_parquet_json:
            persist_dir = os.path.join(SESSION_ROOT, req_id) 
            db_files = [f for f in all_sql_parquet_json if f.filename.lower().endswith(".db", ".sqlite", ".sqlite3", ".duckdb")]
            sql_files = [f for f in all_sql_parquet_json if f.filename.lower().endswith((".sql"))]
            pj_files = [f for f in all_sql_parquet_json if f.filename.lower().endswith((".parquet", ".json"))]
            try:
                ctx = await process_sql_parquet_json(
                task="",                 # we pass it, but your agent will do the thinking later
                db_files=db_files,
                sql_files=sql_files,
                parquet_json_files=pj_files,
                db_urls=[],
                sql_urls=[],
                parquet_json_urls=[],
                persist_dir=persist_dir,
                return_format="json"                   # keep your default _session_sql
                )
                ctx_json = ctx if isinstance(ctx, dict) else json.loads(ctx)
                sql_query = await sql_parquet_json_agent(
                    task_description=task,
                    engine=ctx_json.get("engine"),
                    session_db_path=ctx_json.get("session_db_path"),
                    sample_preview=ctx_json.get("tables")
                )
                exec_output = execute_llm_python(sql_query, session_db_path=ctx_json.get("session_db_path"))
                sql_parquet_json_text = _coerce_to_text(exec_output)
                print(f"[archive_agent] SQL/Parquet/JSON agent len={len(sql_parquet_json_text)}")
            except Exception as e:
                print(f"[archive_agent] SQL/Parquet/JSON agent error: {e}")

    # 4) Return strings ready for your contexts
    return {
        "csv":   csv_text  or "",
        "pdf":   pdf_text  or "",
        "image": image_text or "",
        "html":  html_text or "",
        "sql_parquet_json": sql_parquet_json_text or ""
    }

