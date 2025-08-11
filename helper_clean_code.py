import re
import pandas as pd
# === Code cleaning ===
def clean_code(code: str) -> str:
    """
    Extract only the first Python code block from the LLM output.
    If none found, fallback to returning the whole string (trimmed).
    """
    match = re.search(r"```python(.*?)```", code, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        # fallback: remove leading non-code lines but this is a last resort
        return code.strip()
    
def clean_url(u: str) -> str:
    # strip whitespace first
    u = u.strip()
    # strip trailing quotes (both single and double) and brackets/punctuation
    u = u.strip('"\'')

    # strip other common trailing punctuation/brackets
    u = u.strip('.,;:!?)]}>')

    return u

import json

def ensure_str(data):
    if isinstance(data, (dict, list)):
        # Pretty JSON so it stays easily readable for the LLM
        return json.dumps(data, ensure_ascii=False, indent=2)
    return str(data)  # for markdown, plain text, etc.