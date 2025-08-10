import os, httpx, tempfile, tarfile, re
import pandas as pd
from io import StringIO
import duckdb
import pdfplumber
from typing import List
from helper_html import render_html_url


def get_previews_from_url(text: str) -> List[dict]:
    """
    Extracts structured previews from all URLs (HTTP/S and S3) mentioned in the input text.
    Returns a list of dictionaries with source, columns, sample rows, or errors.
    """

    url_matches = re.findall(r'https?://\S+', text)
    s3_matches = re.findall(r's3://[^\s\'"`]+', text)

    previews = []

    # === Handle HTTPS URLs ===
    for url in url_matches:
        try:
            df = None  # Reset df each time

            if url.endswith(".csv"):
                df = pd.read_csv(url)

            elif url.endswith(".json"):
                resp = httpx.get(url)
                try:
                    df = pd.read_json(StringIO(resp.text), lines=True)
                except:
                    df = pd.json_normalize(resp.json())

            elif url.endswith(".pdf"):
                with pdfplumber.open(httpx.get(url, stream=True).raw) as pdf:
                    text = "\n".join([page.extract_text() or '' for page in pdf.pages])
                previews.append({
                    "source": url,
                    "columns": ["text"],
                    "sample_rows": [{"text": text[:500]}]
                })
                continue  # skip to next URL

            elif url.endswith(".tar.gz") or url.endswith(".tar"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz") as tmp:
                    tmp.write(httpx.get(url).content)
                    tar_path = tmp.name

                with tempfile.TemporaryDirectory() as extract_dir:
                    with tarfile.open(tar_path, "r:*") as tar:
                        tar.extractall(extract_dir)

                    for root, _, files in os.walk(extract_dir):
                        for file in files:
                            fpath = os.path.join(root, file)
                            try:
                                if file.endswith(".csv"):
                                    df = pd.read_csv(fpath)
                                    break
                                elif file.endswith(".json"):
                                    try:
                                        df = pd.read_json(fpath, lines=True)
                                    except:
                                        df = pd.json_normalize(pd.read_json(fpath))
                                    break
                                elif file.endswith(".parquet"):
                                    df = pd.read_parquet(fpath)
                                    break
                            except Exception:
                                continue
                        else:
                            continue
                        break
                    else:
                        previews.append({"source": url, "error": "No usable file found in archive"})
                        continue

            else:
                try:
                    html_text = render_html_url(url)
                    previews.append({
                        "rendered_html": html_text,
                        "source": url
                    })
                    continue
                except Exception as e:
                    previews.append({"source": url, "error": f"Playwright HTML extraction failed: {e}"})
                    continue

            # For all structured formats (CSV, JSON, Parquet)
            if df is not None:
                previews.append({
                    "source": url,
                    "columns": df.columns.tolist(),
                    "sample_rows": df.head(5).to_dict(orient="records")
                })

        except Exception as e:
            previews.append({"source": url, "error": f"Preview failed: {str(e)}"})

    # === Handle S3 URLs ===
    for s3_url in s3_matches:
        try:
            if s3_url.endswith(".parquet"):
                duckdb.sql("INSTALL httpfs; LOAD httpfs; INSTALL parquet; LOAD parquet;")
                df = duckdb.sql(f"SELECT * FROM read_parquet('{s3_url}') LIMIT 5").df()
                previews.append({
                    "source": s3_url,
                    "columns": df.columns.tolist(),
                    "sample_rows": df.head(5).to_dict(orient="records")
                })
            else:
                previews.append({"source": s3_url, "error": "Unsupported S3 file type"})
        except Exception as e:
            previews.append({"source": s3_url, "error": f"S3 Preview failed: {str(e)}"})

    return previews
