import os
import asyncio
from dotenv import load_dotenv
from anthropic import AsyncAnthropic
from PIL import Image
from io import BytesIO
import base64
import httpx
from typing import List, Optional
from fastapi import UploadFile
import mimetypes
import asyncio
from io import BytesIO
from starlette.datastructures import UploadFile as StarletteUploadFile
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

def _downscale_image_bytes(data: bytes, max_side: int = 1400, jpeg_quality: int = 85) -> bytes:
    """Downscale to reduce tokens. If Pillow missing or fails, return original."""
    try:
        im = Image.open(BytesIO(data))
        im = im.convert("RGB")  # normalize
        w, h = im.size
        scale = max(w, h) / float(max_side)
        if scale > 1.0:
            im = im.resize((int(w/scale), int(h/scale)))
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        return buf.getvalue()
    except Exception as e:
        print(f"[image_agent] downscale failed, sending original. err={e}")
        return data

async def image_agent(
    image_files: Optional[List[UploadFile]] = None,
    image_urls: Optional[List[str]] = None,
    task: str = "",
    *,
    max_side: int = 1400,
    jpeg_quality: int = 85,
    model: str = "claude-3-5-sonnet-20241022",
) -> str:
    image_files = image_files or []
    image_urls = image_urls or []

    print(f"[image_agent] START  files={len(image_files)}  urls={len(image_urls)}")

    bytes_items: List[tuple[str, bytes, str]] = []

    # URLs → bytes
    for url in image_urls:
        try:
            r = httpx.get(url, timeout=30, follow_redirects=True)
            print(f"[image_agent] GET {url} -> {r.status_code}  ct={r.headers.get('content-type')}")
            r.raise_for_status()
            content = r.content
            if not content:
                print(f"[image_agent] EMPTY content from {url}")
                continue
            mime = r.headers.get("content-type") or mimetypes.guess_type(url)[0] or "image/jpeg"
            bytes_items.append((url, content, mime))
        except Exception as e:
            print(f"[image_agent] URL ERROR {url}: {e}")

    # Uploaded files
    for uf in image_files:
        try:
            b = await uf.read()
            name = getattr(uf, "filename", "<upload>")
            mime = getattr(uf, "content_type", None) or mimetypes.guess_type(name)[0] or "image/jpeg"
            if not b:
                print(f"[image_agent] EMPTY upload: {name}")
                continue
            bytes_items.append((name, b, mime))
        except Exception as e:
            print(f"[image_agent] FILE ERROR {getattr(uf,'filename','<upload>')}: {e}")

    if not bytes_items:
        return "No image content available to analyze."

    # Downscale images
    processed: List[tuple[str, bytes, str]] = []
    for label, data, mime in bytes_items:
        orig_len = len(data)
        data2 = _downscale_image_bytes(data, max_side=max_side, jpeg_quality=jpeg_quality)
        if data2 != data:
            print(f"[image_agent] DOWNSCALED {label}: {orig_len} -> {len(data2)} bytes")
            mime = "image/jpeg"
        processed.append((label, data2, mime))

    # Build Anthropic content blocks
    content_blocks = []
    for label, data, mime in processed:
        head = data[:8]
        print(f"[image_agent] HEAD {label}: {head!r}, mime={mime}")
        b64 = base64.b64encode(data).decode("utf-8")
        content_blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime,
                "data": b64
            }
        })

    # Append task text last
    prompt = (
        "You are an image analysis agent. Look only at the provided images.\n"
        "Perform the task precisely. If measurements are unclear, state assumptions.\n"
        "Return factual text only.\n\n"
        f"Task: {task}"
    )
    content_blocks.append({"type": "text", "text": prompt})

    print(f"[image_agent] BLOCKS images={len(processed)} total_blocks={len(content_blocks)}")

    # Call Anthropic
    resp = await anthropic_client.messages.create(
        model=model,
        max_tokens=1200,
        temperature=0.2,
        messages=[{"role": "user", "content": content_blocks}]
    )

    # Extract answer text
    pieces = []
    for part in (resp.content or []):
        if getattr(part, "type", None) == "text":
            pieces.append(getattr(part, "text", "") or "")
        elif isinstance(part, dict) and part.get("type") == "text":
            pieces.append(part.get("text", "") or "")
    answer = "\n".join(p for p in pieces if p).strip()

    print(f"[image_agent] DONE answer_len={len(answer)}")
    return answer or "[No image answer returned]"

async def main():
    # Local test images (ensure these files exist next to this script)
    img1_path = "wbs.png"
    img2_path = "wbs_timeline.jpg"

    image_files = []
    image_urls = []  # you can add URLs here later if you want

    # Read local images and wrap as Starlette UploadFile objects
    for p in [img1_path, img2_path]:
        if not os.path.exists(p):
            print(f"[test] Missing image file: {p}")
            continue
        with open(p, "rb") as f:
            data = f.read()
        image_files.append(
            StarletteUploadFile(filename=os.path.basename(p), file=BytesIO(data))
        )
        print(f"[test] Prepared upload: {p}, bytes={len(data)}")

    # Task: extract main WBS heads + timeline mapping dates to deliverables
    task = (
        "You will receive images that contain a Work Breakdown Structure (WBS) chart and a timeline.\n"
        "1) Identify the main/top-level heads (phases) of the WBS. Return them as a concise list.\n"
        "2) From the timeline image, extract the major deliverables with their dates.\n"
        "   Return a JSON object with two keys:\n"
        "   {\n"
        "     \"wbs_heads\": [\"Head1\", \"Head2\", ...],\n"
        "     \"timeline\": [\n"
        "        {\"date\": \"YYYY-MM-DD\", \"code\": \"D1\", \"title\": \"<deliverable title>\"},\n"
        "        {\"date\": \"YYYY-MM-DD\", \"code\": \"D2\", \"title\": \"<deliverable title>\"}\n"
        "     ]\n"
        "   }\n"
        "Rules:\n"
        "- If the image shows month/day labels without year, infer year only if obvious; else omit year (use MM-DD).\n"
        "- If a deliverable code is not visible, set \"code\" to null.\n"
        "- If a title is not visible, set \"title\" to null.\n"
        "- Do not include any text outside the JSON."
    )

    print("[test] Calling image_agent ...")
    result = await image_agent(
        image_files=image_files,
        image_urls=image_urls,
        task=task,
        # optional knobs:
        max_side=1400,
        jpeg_quality=85,
    )
    print("\n=== Image Agent Result ===")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())