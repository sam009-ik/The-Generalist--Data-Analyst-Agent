import os
import httpx
import asyncio
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_2")

async def generate_structured_preview(rendered_html: str, task_description: str = "") -> str:
    """
    Uses OpenAI to extract relevant structured content from raw rendered HTML.
    Returns cleaned markdown or CSV-style string.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = f"""
You are an HTML structuring assistant. Your job is to extract useful structured data from raw rendered HTML content.

- Focus on extracting **tables**, **data points**, or **analytical text** relevant for data analysis.
- Output should be a **markdown table**, **CSV**, or **clean structured text block**, not raw HTML.
- If the content contains a table, extract only the most relevant table and convert it to markdown.
- If the page has multiple data sections, summarize only the most relevant one based on the task.
- Do NOT include irrelevant UI text (e.g., navigation, cookie banners, donate buttons).
- The output should be concise, clean, and analysis-ready.

If no structured data is present, summarize the visible textual content that seems most relevant to the task.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Task: {task_description}\n\nHTML:\n{rendered_html}"}
    ]

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1500
    }

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        content = resp.json()['choices'][0]['message']['content'].strip()
        return content
    
 