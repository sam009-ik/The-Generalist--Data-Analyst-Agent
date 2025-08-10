import os, subprocess, tempfile, sys

# === Execute safely ===
def execute_code(code: str, timeout: int = 120):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path], capture_output=True, text=True, timeout=timeout
        )
        return proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return "", "Execution timed out"
    finally:
        os.remove(tmp_path)