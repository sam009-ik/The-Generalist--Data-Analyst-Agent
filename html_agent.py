import os
import asyncio
from dotenv import load_dotenv
from anthropic import AsyncAnthropic

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

async def html_agent(rendered_html: str, task_description: str = "") -> str:
    """
    Uses Claude 3.5 to extract relevant structured content from rendered HTML.
    Returns cleaned markdown, CSV, or structured text.
    """
    system_prompt = """
You are an HTML Specialist Agent.
Do not bother about other files mentioned in the task. Your job is to:
- Keep the task description in mind to understand what data might be relevant from what you do have access to.
- Instead, provide **all relevant data** from the page that might be useful for downstream LLM processing.
- Extract all relevant tables or data blocks and convert each table into markdown or CSV format.
- Do not include raw HTML or any irrelevant UI text (navigation, cookie banners, ads).
- Do not Hallucinate and do not fabricate data.
- Provide clean, parseable output for full data ingestion. """

    user_prompt = f"Task: {task_description}\n\nHTML:\n{rendered_html}"

    response = await anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1500,
        temperature=0.3,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    try:
        return response.content[0].text.strip()
    except Exception as e:
        return f"Error parsing Anthropic response: {e}"
