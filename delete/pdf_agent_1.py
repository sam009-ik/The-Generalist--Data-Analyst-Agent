import base64
import httpx
from typing import List, Optional, Union
from fastapi import UploadFile
from anthropic import AsyncAnthropic
import os
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

async def pdf_agent2(
    pdf_files: Optional[List[UploadFile]] = None,
    pdf_urls: Optional[List[str]] = None,
    task: str = ""
) -> str:
    pdf_files = pdf_files or []
    pdf_urls = pdf_urls or []

    # ---- Build content blocks: text first ----
    content_blocks = [{
        "type": "text",   # you can also use "input_text"
        "text": (
            "You are a PDF extraction agent. Read only the provided PDFs and extract "
            "numbers/tables/passages that directly help with the task. "
            "Do not hallucinate and do not answer beyond the PDFs.\n\n"
            f"Task: {task}"
        )
    }]

    # ---- Add document blocks for URL PDFs (no download) ----
    for url in pdf_urls:
        # Claude can fetch public PDFs directly
        content_blocks.append({
            "type": "document",
            "source": {
                "type": "url",
                "url": url
            }
        })
        print(f"[pdf_agent] Added URL doc: {url}")

    # ---- Add document blocks for uploaded PDFs (base64) ----
    for pf in pdf_files:
        try:
            data = await pf.read()
            if not data:
                print(f"[pdf_agent] Empty file: {pf.filename}")
                continue
            b64 = base64.b64encode(data).decode("utf-8")
            content_blocks.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": b64
                }
            })
            print(f"[pdf_agent] Added file doc: {pf.filename}, bytes={len(data)}")
        except Exception as e:
            print(f"[pdf_agent] Error reading {getattr(pf,'filename','<upload>')}: {e}")

    # Guard
    if len(content_blocks) == 1:
        return "No PDF content available to analyze."

    # ---- Call Claude ----
    resp = await anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",   # or your pinned model
        max_tokens=4000,
        temperature=0.2,
        messages=[{"role": "user", "content": content_blocks}]
    )

    # ---- Extract text blocks ----
    out_parts = []
    for part in (resp.content or []):
        # SDK may return objects or dicts—handle both
        if getattr(part, "type", None) == "text":
            out_parts.append(getattr(part, "text", "") or "")
        elif isinstance(part, dict) and part.get("type") == "text":
            out_parts.append(part.get("text", "") or "")
    extracted = "\n".join(p.strip() for p in out_parts if p.strip())
    return extracted or "[No extractable text returned from PDF agent]"
