import os, httpx
from dotenv import load_dotenv
import io, contextlib, duckdb, pandas as pd, numpy as np
from process_sql_parquet_json import process_sql_parquet_json
import asyncio
import json
from starlette.datastructures import UploadFile as StarletteUploadFile
from typing import List, Optional
from fastapi import UploadFile

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_2")

async def sql_parquet_json_agent(task_description: str, engine: str, session_db_path: str, sample_preview):
    """
    Returns a single Python script (as a string). You will execute it locally.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    tables_list = sample_preview.get("tables", sample_preview) if isinstance(sample_preview, dict) else sample_preview
    allowed_tables = [t["name"] for t in tables_list]

    system_prompt = f"""
You are a agent that deals with SQL-PARQUET-JSON files and writes ONLY Python code (no prose, no backticks).
Context:
- ENGINE: {engine}
- SESSION_DB_PATH: {session_db_path}
- ALLOWED_TABLES: {allowed_tables}\n"
- SAMPLE_PREVIEW_DATA: {sample_preview}

Rules:
- Output a complete, executable Python script ONLY.
- Use: 
  import duckdb, pandas as pd, numpy as np
- Connect with:
  con = duckdb.connect(SESSION_DB_PATH, read_only=True)
- USE ONLY the registered tables listed in ALLOWED_TABLES.
- NEVER read external files (no '...parquet' paths, no read_parquet, no FROM 'path').
- Identifiers may include dots/spaces; always double-quote them in SQL, or create a cleaned view with underscore names first
- Explore safely: start with small LIMITs; use df = con.execute(SQL).df()
- Do NOT use the network or read/write local files. Work in-memory only.
- Cast types carefully (e.g., CAST(col AS DOUBLE)), parse dates with DuckDB or pandas.
- If analysis requires ML/stats (regression, classification, clustering, time series), use numpy/pandas and scikit-learn if available; if sklearn is missing, implement a simple alternative with numpy (e.g., OLS via np.linalg.lstsq).
- At the end, PRINT a concise markdown answer.
  - If useful, also print a small markdown table via df.head(20).to_markdown(index=False).
- Never reference columns that don’t exist; rely on SAMPLE_PREVIEW_TABLES and validate with quick SELECT * LIMIT 5.
- No GUI plotting. If you must plot, skip showing/saving and instead print key numeric results.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Task: {task_description}\nReturn ONLY Python code."}
    ]

    payload = {
        "model": "gpt-4o-mini",     # or keep "gpt-3.5-turbo" if you must
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2000
    }

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()



def execute_llm_python(code_str: str, session_db_path: str):
    # Inject safe globals; give the script SESSION_DB_PATH + common libs
    globs = {
        "duckdb": duckdb,
        "pd": pd,
        "np": np,
        "SESSION_DB_PATH": session_db_path,
    }
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code_str, globs, {})
        out = buf.getvalue().strip()
        return {"ok": True, "stdout": out}
    except Exception as e:
        return {"ok": False, "error": str(e), "stdout": buf.getvalue().strip()}

async def main():
    # 0) read the single task once
    with open("question12.txt", "r", encoding="utf-8") as f:
        task = f.read().strip()

    # 1) build the session from your uploaded files (example with iris.parquet)
    #    add more UploadFile(...) items if you want to include more parquet/json/sqlite/etc
    parquet_path = "sakila.db"  # change if needed
    with open(parquet_path, "rb") as f:
        uf = UploadFile(filename=os.path.basename(parquet_path), file=io.BytesIO(f.read()))

    ctx = await process_sql_parquet_json(
        task="",  # we don’t use task here
        db_files=[uf],
        parquet_json_files=[],
        persist_dir=None,
        return_format="json"
    )
    ctx_json = json.loads(ctx) if isinstance(ctx, str) else ctx
    print(f"Output of process_sql_parquet_json Context: {ctx_json}")
    engine = ctx_json["engine"]
    session_db_path = ctx_json["session_db_path"]
    sample_preview = ctx_json["tables"]  # compact table + columns + small samples

    # 2) ask LLM for a SINGLE python program (no prose) to solve the task
    #    your sql_parquet_json_agent should instruct: “python-only, no markdown, use SESSION_DB_PATH for duckdb.connect(...), print results”
    code = await sql_parquet_json_agent(
        task_description=task,
        engine=engine,
        session_db_path=session_db_path,
        sample_preview=sample_preview
    )

    print("\n================ GENERATED PYTHON ================\n")
    print(code[:2000] + ("\n... [truncated]" if len(code) > 2000 else ""))

    # 3) execute ONCE and print the output
    print("\n================ EXECUTION OUTPUT ================\n")
    result = execute_llm_python(code, session_db_path=session_db_path)
    if result["ok"]:
        print(result["stdout"])
    else:
        print("ERROR:", result["error"])
        if result["stdout"]:
            print("\nPartial stdout:\n", result["stdout"])

if __name__ == "__main__":
    asyncio.run(main())