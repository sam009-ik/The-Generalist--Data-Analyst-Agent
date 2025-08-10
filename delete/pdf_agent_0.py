import fitz
from tabula import read_pdf
from dotenv import load_dotenv
import os
import asyncio
from dotenv import load_dotenv
from anthropic import AsyncAnthropic
import base64
from io import BytesIO
from fastapi import UploadFile
from typing import List, Union
import httpx
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

async def pdf_agent(pdfs: List[Union[UploadFile]], task_description: str = "") -> str:
    """
    Sends uploaded PDF files + a task description to Claude and gets back structured relevant data.
    Accepts a list of Starlette UploadFile objects.
    """
    all_pdfs = []
    for pdf in pdfs:
        if isinstance(pdf, str):
            if pdf.startswith("http://") or pdf.startswith("https://"):
                response = httpx.get(pdf)
                if response.status_code != 200:
                    raise ValueError(f"Failed to download PDF from URL: {pdf}")
                file_bytes = response.content
            else:
                with open(pdf, "rb") as f:
                    file_bytes = f.read()
        else:
            file_bytes = await pdf.read()

        pdf_data_b64 = base64.b64encode(file_bytes).decode("utf-8")
        all_pdfs.append(pdf_data_b64)

    # Claude prompt
    user_prompt = f"""
You are a PDF Specialist Agent. Only deal with pdf files based on the task and do not bother about other files. Your job is to:
- Read the entire PDFs (or relevant pages)
- Identify tables, data, or explanations that help answer the task
- Extract them clearly without any extra natural language commentary.
- Do NOT hallucinate or fabricate data
- Do not attempt to answer the final question — just extract all useful information for downstream LLM processing.

Task: {task_description}
"""
    # Build one document block per PDF
    content_blocks = [
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": pdf_data
            }
        }
        for pdf_data in all_pdfs
    ]

    # Add the user prompt as the final message content block
    content_blocks.append({
        "type": "text",
        "text": user_prompt
    })

    # Send to Claude
    response = await anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4000,
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": content_blocks  
            }
        ]
    )

    try:
        return response.content[0].text
    except Exception as e:
        print("Claude response error:", e)
        print("Full response object:", response)
        raise

