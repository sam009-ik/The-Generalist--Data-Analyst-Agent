from pprint import pprint
import re
import uuid
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os, json, asyncio, sys
import pandas as pd
from io import StringIO
from starlette.datastructures import UploadFile as StarletteUploadFile
from typing import List
import pprint
# ========== OWN FUNCTIONS ==========
#from html_structuring_agent import generate_structured_preview
from helper_html import render_html_file, render_html_url
from helper_clean_code import clean_code, clean_url
from helper_execute_code import execute_code
# Anthropic SDK
from anthropic import AsyncAnthropic
from html_agent import html_agent
from pdf_agent import pdf_agent 
from csv_tsv_xlsx_agent import csv_tsv_xlsx_agent #Using Powerdrill
from image_agent import image_agent
from archive_agent import archive_agent
from sql_parquet_json_agent import sql_parquet_json_agent, execute_llm_python
from process_sql_parquet_json import process_sql_parquet_json

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SESSION_ROOT = os.getenv("SESSION_ROOT", "/data/_session_sql")  # use /data on Render; falls back locally

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

class AnalysisRequest(BaseModel):
    question: str

# === Helper: Call Anthropic to generate Python ===
async def data_analyst_agent(task: str, html_context=None, pdf_context=None, csv_tsv_xlsx_context=None, image_context=None, archive_context=None, sql_parquet_json_context=None) -> str:
    #data_source = preview.get("source", "")
    system_prompt = fr"""You are a skilled data analyst writing complete, safe, and clean Python code to solve the user's data analysis task.
<General Instructions>
    Based on the user's main task generate **only valid Python code** that:
        - Reads the complete data correctly based on the task (from HTML tables, CSV URLs, PDFs or S3 parquet files via DuckDB).
        - Handles messy data robustly using pandas and standard libraries.
        - Before using `json.dumps(...)`, ensure all numeric values (like sums, means, etc.) are explicitly cast to native Python types using `int(...)` or `float(...)`. This avoids TypeError from numpy/pandas types.
        - When cleaning monetary values (like gross revenues), use .replace(r'[^\d.]', '', regex=True) to strip out all non-numeric characters except the decimal point
        - Always use pd.to_numeric(..., errors='coerce') for numeric conversion. Never use .astype(float) on user data — this will cause crashes on malformed entries like "T$...".
        - In case the question text involves a date, ensure you take that into consideration and answer accordingly.
        - Drops missing values using `df.dropna(subset=[...])` before running any model fitting or mathematical operations.
        - Always validates that both X and y (for regression) are numeric and contain no NaNs before calling `.fit()`.
        - Uses raw strings for regex (e.g., `r'\\$|,'`) to avoid escape errors.
        - When using `pd.read_html()` with a BeautifulSoup tag or HTML string, wrap it as `StringIO(str(tag))` and import `from io import StringIO`.
        - Imports **all required packages** at the top of the script, including `json`, `io`, `base64`, `matplotlib.pyplot as plt`, etc.
        - Does **not** write to files or show plots — encode any plots using base64 if requested.
</General Instructions>
<Plotting Instructions>
    If the task involves plotting (e.g., scatterplot, regression line, barplot):
        - Use `io.BytesIO()` to capture the figure in memory.
        - Save with `plt.savefig(buf, format='png')`, followed by `buf.seek(0)`.
        - Encode using `base64.b64encode(buf.read()).decode('utf-8')`.
        - Format as a data URI like "data:image/png;base64,<...>".
        - Insert the base64 string in the correct position in the final result — i.e., exactly where the corresponding question is.
</Plotting Instructions>

The final output must be a single line:
```python
print(json.dumps([...])))
"""
    if html_context or (archive_context and archive_context[0]["content"].get("html")):
        html_text = ""
        if html_context:
            html_text += f"\nFrom Direct HTML Source: {html_context['source']}\n{html_context['content']}\n"
        if archive_context and archive_context[0]["content"].get("html"):
            html_text += f"\nFrom Archive HTML:\n{archive_context[0]['content']['html']}\n"
        system_prompt += fr"""<HTML Instructions>
    Here is the relevant HTML content from the `HTML Specialist Agent` which has already been extracted and structured for you.
    You do **not** need to re-fetch or parse the HTML yourself.
    Focus on using the provided structured html and then answer the questions in the format requested.
    </HTML Instructions>
    <Structured HTML>
        {html_text}
    </Structured HTML>
    """
    if pdf_context or (archive_context and archive_context[0]["content"].get("pdf")):
        pdf_text = ""
        if pdf_context:
            pdf_text += f"\nFrom Direct PDF Source: {pdf_context['source']}\n{pdf_context['content']}\n"
        if archive_context and archive_context[0]["content"].get("pdf"):
            pdf_text += f"\nFrom Archive PDF:\n{archive_context[0]['content']['pdf']}\n"
            system_prompt += f""" <PDF Instructions>
    Here is the relevant content for the task from the `PDF Specialist Agent` which has already been extracted and structured for you.
    You do **not** need to re-fetch or parse the pdf yourself.
    Focus on using the provided structured pdf data and then answer the questions in the format requested.
    </PDF Instructions>
    <Structured PDF>
        {pdf_text}
    </Structured PDF>
    """
    if csv_tsv_xlsx_context or (archive_context and archive_context[0]["content"].get("csv")):
        csv_text = ""
        if csv_tsv_xlsx_context:
            csv_text += f"\nFrom Direct CSV/Excel Source: {csv_tsv_xlsx_context['source']}\n{csv_tsv_xlsx_context['content']}\n"
        if archive_context and archive_context[0]["content"].get("csv"):
            csv_text += f"\nFrom Archive CSV/Excel:\n{archive_context[0]['content']['csv']}\n"
        system_prompt += f"""<CSV-TSV-EXCEL Instructions>
    Here is the relevant content for the task from the `csv-tsv-xlsx Specialist Agent` which has already been extracted and structured for you.
    You do **not** need to re-fetch or parse the csv/tsv/excel yourself.
    Focus on using the provided structure and then answer the questions in the format requested.
    </CSV-TSV-EXCEL Instructions>
    <CSV-TSV-EXCEL Data>
        {csv_text}
    </CSV-TSV-EXCEL Data>
    """
    if image_context or (archive_context and archive_context[0]["content"].get("image")):
        img_text = ""
        if image_context:
            img_text += f"\nFrom Direct Image Source: {image_context['source']}\n{image_context['content']}\n"
        if archive_context and archive_context[0]["content"].get("image"):
            img_text += f"\nFrom Archive Image:\n{archive_context[0]['content']['image']}\n"
        system_prompt += f"""<Image Instructions>
    Here is the relevant content for the task from the `Image Specialist Agent` which has already been extracted and structured for you.
    You do **not** need to re-fetch or parse the image yourself.
    Focus on using the provided structure and then answer the questions in the format requested.
    <Image Data>
        {img_text}
    </Image Data> """
    if sql_parquet_json_context:
        system_prompt += f"""<SQL-Parquet-JSON Instructions>
    Here is the relevant content for the task from the `SQL/Parquet/JSON Specialist Agent` which has already been extracted and structured for you.
    You do **not** need to re-fetch or parse the SQL/Parquet/JSON yourself.
    Focus on using the provided structure and then answer the questions in the format requested.
    </SQL-Parquet-JSON Instructions>
    <Structured SQL-Parquet-JSON>
        {f"\nRelevant Data: {sql_parquet_json_context[0]['source']}\n{sql_parquet_json_context[0]['content']}\n"}
    </Structured SQL-Parquet-JSON>
    """

    response = await anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1500,
        temperature=0.2,
        system=system_prompt,
        messages=[
            {"role": "user", "content": task}
        ]
    )

    return response.content[0].text.strip()


# === Main Endpoint ===
@app.post("/api/")
async def analyze(request: Request):
    stdout = ""
    stderr = ""
    try:
        form = await request.form()
        question_file = form.get("questions.txt")
        if not question_file:
            return {"error": "Missing 'questions.txt' in request."}
        # Read question
        question_text = (await question_file.read()).decode("utf-8").strip()
        url_matches = re.findall(r'https?://[^\s"\'>]+', question_text)
        url_matches = [clean_url(u) for u in url_matches]
        print("Cleaned URLs Found: ", url_matches)
        # Uploaded files
        other_files = [
            v for k, v in form.items()
            if isinstance(v, StarletteUploadFile) and k != "questions.txt"
        ]
        print("Files found:", [f.filename for f in other_files])
        # Context holders
        html_context = []
        pdf_context = []
        csv_tsv_xlsx_context = []
        image_context = []
        archive_context = None
        sql_parquet_json_context = None
        
        # === Categorize by type ===
        DB_EXTS  = (".db", ".sqlite", ".sqlite3", ".duckdb")
        SQL_EXTS = (".sql",)
        PJ_EXTS  = (".parquet", ".json")
            # ========= FILES =========
        html_files = [f for f in other_files if f.filename.endswith(".html")]
        print("HTML Files:", [f.filename for f in html_files])
        pdf_files = [f for f in other_files if f.filename.endswith(".pdf")]
        print("PDF Files:", [f.filename for f in pdf_files])
        csv_tsv_xlsx_files = [f for f in other_files if f.filename.lower().endswith((".csv", ".tsv", ".xlsx"))]
        print("CSV/TSV/XLSX Files:", [f.filename for f in csv_tsv_xlsx_files])
        image_files = [f for f in other_files if f.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        print("Image Files:", [f.filename for f in image_files])
        archive_files = [f for f in other_files if f.filename.lower().endswith((".zip", ".tar", ".tar.gz"))]
        print("Archive Files:", [f.filename for f in archive_files])
        db_files   = [f for f in other_files if f.filename.lower().endswith(DB_EXTS)]
        print("DB Files:", [f.filename for f in db_files])
        sql_files  = [f for f in other_files if f.filename.lower().endswith(SQL_EXTS)]
        print("SQL Files:", [f.filename for f in sql_files])
        pj_files   = [f for f in other_files if f.filename.lower().endswith(PJ_EXTS)]
        print("Parquet/JSON Files:", [f.filename for f in pj_files])
            # ========= URLS =========
        html_urls = [
            url for url in url_matches
            if not url.lower().endswith((".pdf", ".csv", ".tsv", ".xlsx", ".json", ".png", ".jpg", ".jpeg", ".webp"))
        ]
        print("HTML URLs:", html_urls)
        pdf_urls = [url for url in url_matches if url.lower().endswith(".pdf")]
        print("PDF URLs:", pdf_urls)
        csv_tsv_xlsx_urls = [
            url for url in url_matches
            if url.lower().endswith((".csv", ".tsv", ".xlsx"))
        ]
        print("CSV/TSV/XLSX URLs:", csv_tsv_xlsx_urls)
        image_urls = [u for u in url_matches if u.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        print("Image URLs:", image_urls)
        archive_urls = [u for u in url_matches if u.lower().endswith((".zip", ".tar", ".tar.gz"))]
        print("Archive URLs:", archive_urls)
        db_urls = [u for u in url_matches if u.lower().endswith(DB_EXTS)]
        print("DB URLs:", db_urls)
        sql_urls = [u for u in url_matches if u.lower().endswith(SQL_EXTS)]
        print("SQL URLs:", sql_urls)
        pj_urls  = [u for u in url_matches if u.lower().endswith(PJ_EXTS)]
        print("Parquet/JSON URLs:", pj_urls)
        # === Handle HTML ===
        if html_files or html_urls:
            print("Processing HTML files or URLs...")
            rendered_html_file = ""
            rendered_html_urls = ""

            if html_files:
                rendered_html_file = await render_html_file(html_files)
                print("Rendered HTML from files:", rendered_html_file)
            if html_urls:
                rendered_html_urls = await asyncio.to_thread(render_html_url, html_urls)
                print("Rendered HTML from URLs:", rendered_html_urls)

            full_html = (rendered_html_file or "") + (rendered_html_urls or "")
            if full_html.strip():
                structured_html = await html_agent(full_html, question_text)
                print("Content from HTML Agent:", structured_html)
                html_context.append({
                    "source": {
                        "HTML Files": [f.filename for f in html_files],
                        "HTML URLs": html_urls
                    },
                    "content": structured_html
                })
        else:
            html_context = None
        # === Handle PDF ===
        if pdf_files or pdf_urls:
            print("Processing PDF files or URLs...")
            # all_pdfs = pdf_files + pdf_urls
            useful_pdf_content = await pdf_agent(pdf_files, pdf_urls, question_text)
            print("Content from PDF Agent:", useful_pdf_content)
            pdf_context.append({
                "source": {
                    "PDF Files": [f.filename for f in pdf_files],
                    "PDF URLs": pdf_urls
                },
                "content": useful_pdf_content
            })
        else:
            pdf_context = None

            # === HANDLE CSV/TSV/XLSX ===
        if csv_tsv_xlsx_files or csv_tsv_xlsx_urls:
            print("Processing CSV/TSV/XLSX files or URLs...")
            # IMPORTANT: correct argument order & types
            useful_csv_content = await csv_tsv_xlsx_agent(
                task_description=question_text,
                uploaded_files=csv_tsv_xlsx_files,
                file_urls=csv_tsv_xlsx_urls
            )
            print("Content from CSV/TSV/XLSX Agent:", useful_csv_content)
            csv_tsv_xlsx_context.append({
                "source": {
                    "csv_files": [f.filename for f in csv_tsv_xlsx_files],
                    "csv_urls": csv_tsv_xlsx_urls
                },
                "content": useful_csv_content or ""
            })
        else:
            csv_tsv_xlsx_context = None
        
        if image_files or image_urls:
            img_result = await image_agent(image_files=image_files, image_urls=image_urls, task=question_text)
            image_text = img_result
            print("Content from Image Agent:", image_text)
            image_context = [{
                "source": {
                    "image_files": [f.filename for f in image_files],
                    "image_urls": image_urls
                },
                "content": image_text
            }]
        else:
            image_context = []
        
        if archive_files or archive_urls:
            print("Processing archive files or URLs...")
            archive_result = await archive_agent(
                archive_files=archive_files,
                archive_urls=archive_urls,
                task=question_text
            )
            print("Content from Archive Agent:", archive_result)
            archive_context = [{
                "source": {
                    "archive_files": [f.filename for f in archive_files],
                    "archive_urls": archive_urls
                },
                "content": archive_result
            }]
        else:
            archive_context = None

        req_id = uuid.uuid4().hex
        persist_dir = os.path.join(SESSION_ROOT, req_id) 
        # === Handle SQL/Parquet/JSON ===
        if db_files or sql_files or pj_files or db_urls or sql_urls or pj_urls:
            print("Processing SQL/Parquet/JSON…")
            # 1 Build the DuckDB session and preview via your existing processor
            ctx = await process_sql_parquet_json(
                task="",                 # we pass it, but your agent will do the thinking later
                db_files=db_files,
                sql_files=sql_files,
                parquet_json_files=pj_files,
                db_urls=db_urls,
                sql_urls=sql_urls,
                parquet_json_urls=pj_urls,
                persist_dir=persist_dir,
                return_format="json"                   # keep your default _session_sql
            )
            # ctx can be dict or JSON string depending on your implementation
            ctx_json = ctx if isinstance(ctx, dict) else json.loads(ctx)
            print("Output of process_sql_parquet_json Context:", ctx_json)

            # 2 Ask your SQL/Parquet/JSON agent to produce Python code for THIS task+context
            #    Keep the prompt simple; the agent writes all code in Python and uses DuckDB session_db_path.
            generated_code = await sql_parquet_json_agent(
                task_description=question_text,
                engine=ctx_json.get("engine"),
                session_db_path=ctx_json.get("session_db_path"),
                sample_preview=ctx_json.get("tables"),
            )
            print("\n================ GENERATED PYTHON ================\n")
            print(generated_code)
            print("\n==================================================\n")

            # 3. Execute the generated Python safely (your existing runner)
            exec_output = execute_llm_python(generated_code, session_db_path=ctx_json.get("session_db_path"))
            print("\n================ EXECUTION OUTPUT ================\n")
            print(exec_output)
            print("\n==================================================\n")

            # 4. Park result for the master agent
            sql_parquet_json_context = [{
                "source": {
                    "db_files":   [f.filename for f in db_files],
                    "sql_files":  [f.filename for f in sql_files],
                    "pj_files":   [f.filename for f in pj_files],
                    "db_urls":    db_urls,
                    "sql_urls":   sql_urls,
                    "pj_urls":    pj_urls,
                },
                "context": ctx_json,           # so master can see session_db_path/tables if needed
                "generated_code": generated_code,
                "content": exec_output or ""   # what we’ll hand to the master agent
            }]
        else:
            sql_parquet_json_context = None

        # Call Data Analyst Agent
        orchestrator = await data_analyst_agent(
            task=question_text,
            html_context=html_context,
            pdf_context=pdf_context,
            csv_tsv_xlsx_context=csv_tsv_xlsx_context,
            image_context=image_context,
            archive_context=archive_context,
            sql_parquet_json_context=sql_parquet_json_context
        )
        print("Master Data Analyst Code:", orchestrator)
        # Clean and execute the code
        cleaned = clean_code(orchestrator)
        stdout, stderr = execute_code(cleaned)

        last_line = stdout.strip().splitlines()[-1]
        data_analyst_ans = json.loads(last_line)

        return data_analyst_ans

    except Exception as e:
        return {
            "error": "Execution failed",
            "details": str(e),
            "stdout": stdout,
            "stderr": stderr
        }
