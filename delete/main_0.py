from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
import os, httpx, subprocess, tempfile, sys, json, re
import pandas as pd
from io import StringIO
import duckdb
import base64
import pdfplumber
import re, os, tempfile, tarfile, zipfile
from typing import List
from starlette.datastructures import UploadFile as StarletteUploadFile
import asyncio
import sys

#======================== OWN FUNCTIONS ========================
from delete.html_structuring_agent import generate_structured_preview
from delete.preview_from_url import get_previews_from_url
from delete.preview_from_file import try_preview_from_files
from helper_html import render_html_from_url
from helper_clean_code import clean_code
from helper_execute_code import execute_code

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_2") # Use the second key for better performance

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    question: str


# === Helper: Call LLM to generate Python ===
async def generate_code(task: str, preview: dict, html=None):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data_source = preview.get("source", "")
    system_prompt = fr"""You are a skilled data analyst writing complete, safe, and clean Python code to solve the user's data analysis task.
    <Instructions>
Based on the user's task, the main data source {data_source}, and/or the structured html {html} if it exists.
Generate **only valid Python code** that:
- Reads the complete data correctly based on the task (from HTML tables, CSV URLs, or S3 parquet files via DuckDB).
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

If the task involves plotting (e.g., scatterplot, regression line, barplot):

- Use `io.BytesIO()` to capture the figure in memory.
- Save with `plt.savefig(buf, format='png')`, followed by `buf.seek(0)`.
- Encode using `base64.b64encode(buf.read()).decode('utf-8')`.
- Format as a data URI like `"data:image/png;base64,<...>"`.
- Insert the base64 string in the correct position in the final result — i.e., exactly where the corresponding question is.

The final output must be a single line:
```python
print(json.dumps([...]))
</Instructions>
<Sample Schema>
MAIN DATA SOURCE: 
{f"Source: {preview['source']}" if 'source' in preview else ""}
Data preview:
Columns: {preview.get("columns")}
Sample rows:
{json.dumps(preview.get("sample_rows", []), indent=2)}
"""
    if html:
        system_prompt += f"""

        <Additional Context>
        The JavaScript-rendered HTML content from the webpage has already been extracted and structured for you.
        You do **not** need to re-fetch or parse the HTML yourself.

        Instead of trying to extract tables or text again, simply use the data provided in the as structured html (from another llm) as your DataFrame.

        **Do not** call `pd.read_html()` or use `BeautifulSoup` — this has already been done.
        
        """

    payload = {
        "model": "chatgpt-4o-latest",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task}
        ],
        "max_tokens": 1500,
        "temperature": 0.2
    }
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content'].strip()


 
# === Main Endpoint ===
@app.post("/api/")
async def analyze(request: Request):
    stdout = ""
    stderr = ""
    try:
        # 1. Parse the multipart form data
        form = await request.form()

        # 2. Identify the question file (must be questions.txt)
        question_file = form.get("questions.txt")
        if not question_file:
            return {"error": "Missing 'questions.txt' in request."}
        question_text = (await question_file.read()).decode("utf-8").strip()

        # 3. Preview from question content (URL/S3)
        preview = await asyncio.to_thread(get_preview_from_url, question_text)
        print(f"Source: {preview.get('source', '[no source]')}")
        #print("Preview of data: ", preview)

        # 4. All other form values that are UploadFiles = attachments
        other_files = [
            v for k, v in form.items()
            if isinstance(v, StarletteUploadFile) and k != "questions.txt"
        ]

        if other_files:
            for f in other_files:
                if f.filename.endswith(".pdf"):
                    relevant_pdf_content = await pdf_agent(f, question_text)
                    print(relevant_pdf_content)

        if "rendered_html" in preview:
            structured_preview = await generate_structured_preview(preview["rendered_html"], question_text)
            llm_code = await generate_code(question_text, preview, html=structured_preview)
        else:
            llm_code = await generate_code(question_text, preview, html=None)
        print("Code by LLM:", llm_code)
        cleaned = clean_code(llm_code)
        stdout, stderr = execute_code(cleaned)

        last_line = stdout.strip().splitlines()[-1]
        result = json.loads(last_line)

        return {
            "answer": result,
            "warnings": stderr.strip() if stderr else None
        }

    except Exception as e:
        return {
            "error": "Execution failed",
            "details": str(e),
            "stdout": stdout,
            "stderr": stderr
        }