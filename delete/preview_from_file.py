
import json
import pandas as pd
from io import StringIO
import pdfplumber
import os, tempfile, tarfile, zipfile
from typing import List
from fastapi import UploadFile, File

from helper_clean_code import clean_code

# ============== PREVIEW FROM ATTACHED FILES ========================
def try_preview_from_files(files: List[UploadFile]):
    previews = []

    for file in files:
        try:
            filename = file.filename
            lower = filename.lower()

            # CSV / TSV
            if lower.endswith(".csv") or lower.endswith(".tsv"):
                file.file.seek(0)
                df = pd.read_csv(file.file, sep='\t' if lower.endswith(".tsv") else ',')
                previews.append({
                    "filename": filename,
                    "columns": df.columns.tolist(),
                    "sample_rows": df.head(5).to_dict(orient="records")
                })

            # Excel
            elif lower.endswith(".xlsx"):
                file.file.seek(0)
                df = pd.read_excel(file.file)
                previews.append({
                    "filename": filename,
                    "columns": df.columns.tolist(),
                    "sample_rows": df.head(5).to_dict(orient="records")
                })

            # JSON
            elif lower.endswith(".json"):
                file.file.seek(0)
                content = file.file.read().decode("utf-8")
                try:
                    df = pd.read_json(StringIO(content), lines=True)
                except:
                    df = pd.json_normalize(json.loads(content))
                previews.append({
                    "filename": filename,
                    "columns": df.columns.tolist(),
                    "sample_rows": df.head(5).to_dict(orient="records")
                })

            # Parquet
            elif lower.endswith(".parquet"):
                file.file.seek(0)
                df = pd.read_parquet(file.file)
                previews.append({
                    "filename": filename,
                    "columns": df.columns.tolist(),
                    "sample_rows": df.head(3).to_dict(orient="records")
                })

            # HTML table
            elif lower.endswith(".html"):
                file.file.seek(0)
                content = file.file.read().decode("utf-8")
                tables = pd.read_html(StringIO(content))
                if tables:
                    df = tables[0]
                    previews.append({
                        "filename": filename,
                        "columns": df.columns.tolist(),
                        "sample_rows": df.head(3).to_dict(orient="records")
                    })

            # PDF text
            elif lower.endswith(".pdf"):
                file.file.seek(0)
                tables_preview = []
                with pdfplumber.open(file.file) as pdf:
                    text = "\n".join([page.extract_text() or '' for page in pdf.pages])
                    for page_num, page in enumerate(pdf.pages):
                        tables = page.extract_tables()
                        for idx, table in enumerate(tables):
                            header=table[0] if table else []
                            rows = table[1:6]  # Get first 5 rows
                            tables_preview.append({
                                "table_index": idx + 1,
                                "columns": header,
                                "sample_rows": rows
                            })
                preview = {
                    "filename": filename,
                    "columns": ["text"],
                    "sample_rows": [{"text": text[:500]}]
                }
                if tables_preview:
                    preview["tables_preview"] = tables_preview
                previews.append(preview)

            else:
                previews.append({"filename": filename, "error": "Unsupported format for preview"})

        except Exception as e:
            previews.append({"filename": filename, "error": str(e)})

    return previews if previews else None
