import base64
from dotenv import load_dotenv
import httpx
from typing import List, Optional
from fastapi import UploadFile
from anthropic import AsyncAnthropic
import os

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


async def pdf_agent(
    pdf_files: Optional[List[UploadFile]] = None,
    pdf_urls: Optional[List[str]] = None,
    task: str = ""
) -> str:
    pdf_files = pdf_files or []
    pdf_urls = pdf_urls or []

    # ---- collect PDFs as base64 (urls -> download -> base64, files -> read -> base64) ----
    b64_docs: List[str] = []

    # URLs -> base64
    for url in pdf_urls:
        try:
            r = httpx.get(url, timeout=30, follow_redirects=True)
            r.raise_for_status()
            content = r.content
            if not content:
                print(f"[pdf_agent2] Empty response: {url}")
                continue
            b64_docs.append(base64.b64encode(content).decode("utf-8"))
            print(f"[pdf_agent2] URL ok: {url}, bytes={len(content)}")
        except Exception as e:
            print(f"[pdf_agent2] URL error {url}: {e}")

    # Uploads -> base64
    for pf in pdf_files:
        try:
            data = await pf.read()
            if not data:
                print(f"[pdf_agent2] Empty file: {getattr(pf, 'filename', '<upload>')}")
                continue
            b64_docs.append(base64.b64encode(data).decode("utf-8"))
            print(f"[pdf_agent2] File ok: {getattr(pf,'filename','<upload>')}, bytes={len(data)}")
        except Exception as e:
            print(f"[pdf_agent2] File error {getattr(pf,'filename','<upload>')}: {e}")

    if not b64_docs:
        return "No PDF content available to analyze."

    # ---- build a single Anthropic request: text + document blocks (all base64) ----
    content_blocks = [
        {
            "type": "text",
            "text": (
                "You are a PDF extraction agent. Read only the provided PDFs and extract "
                "numbers/tables/passages that directly help with the task. "
                "Do not hallucinate and do not answer beyond the PDFs.\n\n"
                f"Task: {task}"
            )
        }
    ] + [
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": b64,
            }
        }
        for b64 in b64_docs
    ]

    resp = await anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",  # keep your pinned model
        max_tokens=2000,
        temperature=0.2,
        messages=[{"role": "user", "content": content_blocks}],
    )

    # ---- pull only text parts ----
    pieces = []
    for part in resp.content or []:
        if getattr(part, "type", None) == "text":
            pieces.append(getattr(part, "text", "") or "")
        elif isinstance(part, dict) and part.get("type") == "text":
            pieces.append(part.get("text", "") or "")

    return "\n".join(p for p in pieces if p).strip() or "[No extractable text returned]"
