import asyncio
import os
import requests
from io import BytesIO
from typing import List
from fastapi import UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile
from dotenv import load_dotenv
import time

load_dotenv()

POWERDRILL_USERID = os.getenv("POWERDRILL_USER")
POWERDRILL_KEY = os.getenv("POWERDRILL_KEY")

BASE_URL = "https://ai.data.cloud/api/v2/team"
UPLOAD_URL = f"{BASE_URL}/file/upload-datasource"
DATASET_URL = f"{BASE_URL}/datasets"
SESSION_URL = "https://ai.data.cloud/api/v2/team/sessions"
JOB_URL = f"{BASE_URL}/jobs"


def wait_for_dataset_synced(dataset_id: str, timeout_sec: int = 180, poll_every: float = 2.0) -> bool:
    # PD exposes dataset status under v1
    status_url = f"https://ai.data.cloud/api/v1/team/datasets/{dataset_id}/status"
    start = time.time()
    while True:
        r = requests.get(status_url, headers=headers)
        #print("Dataset status:", r.status_code, r.text)
        r.raise_for_status()
        js = r.json()
        data = js.get("data", {}) if isinstance(js, dict) else {}
        invalid = data.get("invalidCount")
        synching = data.get("synchingCount")
        # Done when nothing is invalid and nothing is still syncing
        if invalid == 0 and synching == 0:
            #print("✅ Dataset fully synced.")
            return True
        if time.time() - start > timeout_sec:
            #print("⏰ Timed out waiting for dataset sync.")
            return False
        time.sleep(poll_every)

def extract_answer_and_sources(job_json: dict) -> dict:
    """
    Powerdrill job response -> {
        'answer': <str>,                 # concatenated MESSAGE contents
        'sources': [<str>, <str>, ...],  # just the 'source' strings
    }
    """
    data = (job_json or {}).get("data", {})
    blocks = data.get("blocks", []) if isinstance(data, dict) else []

    # Collect MESSAGE contents
    messages = []

    for b in blocks:
        btype = b.get("type")
        if btype == "MESSAGE":
            text = (b.get("content") or "").strip()
            if text:
                messages.append(text)

    answer = "\n\n".join(messages).strip()
    return {"answer": answer}


headers = {
    "x-pd-api-key": POWERDRILL_KEY,
}

async def csv_tsv_xlsx_agent(
    task_description: str,
    uploaded_files: List[UploadFile] = [],
    file_urls: List[str] = []
) -> str:

    #print("==== Starting Powerdrill Agent ====")

    file_object_keys = []
    dataset_id = None
    datasource_ids = []

    # Step 1: Upload files (from local or URL)
    all_files = []

    for f in uploaded_files:
        content = await f.read()
        all_files.append((f.filename, content, f.content_type or "application/octet-stream"))

    for url in file_urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            filename = url.split("/")[-1]
            content = response.content
            all_files.append((filename, content, "application/octet-stream"))
        except Exception as e:
            print(f"Failed to download from {url}: {e}")

    if not all_files:
        return "❌ No valid files provided."

    # Step 2: Upload files and collect object keys
    for filename, content, content_type in all_files:
        #print(f"Uploading {filename} to Powerdrill...")
        try:
            payload = {"user_id": POWERDRILL_USERID}
            files = [('file', (filename, content, content_type))]
            res = requests.post(UPLOAD_URL, headers=headers, data=payload, files=files)
            #print("Upload Response:", res.status_code, res.text)
            res.raise_for_status()
            res_json = res.json()
            file_object_key = res_json.get("data", {}).get("file_object_key")
            if file_object_key:
                file_object_keys.append((filename, file_object_key))
        except Exception as e:
            print(f"❌ Upload failed for {filename}: {e}")

    if not file_object_keys:
        return "❌ Upload failed for all files."

    # Step 3: Create Dataset
    try:
        dataset_name = "PowerdrillAgentDataset"
        dataset_payload = {
            "name": dataset_name,
            "user_id": POWERDRILL_USERID
        }
        dataset_res = requests.post(DATASET_URL, headers=headers, json=dataset_payload)
        #print("Dataset creation response:", dataset_res.status_code, dataset_res.text)
        dataset_res.raise_for_status()
        dataset_id = dataset_res.json().get("data", {}).get("id")
    except Exception as e:
        return f"❌ Dataset creation failed: {e}"

    # Step 4: Create Data Sources using file_object_keys
    for filename, object_key in file_object_keys:
        try:
            source_payload = {
                "name": filename,
                "type": "FILE",
                "user_id": POWERDRILL_USERID,
                "file_object_key": object_key
            }
            ds_url = f"{DATASET_URL}/{dataset_id}/datasources"
            ds_res = requests.post(ds_url, headers=headers, json=source_payload)
            #print(f"Data source creation for {filename}: {ds_res.status_code}, {ds_res.text}")
            ds_res.raise_for_status()
            datasource_id = ds_res.json().get("data", {}).get("id")
            datasource_ids.append(datasource_id)
        except Exception as e:
            print(f"❌ Data source creation failed for {filename}: {e}")
    if not wait_for_dataset_synced(dataset_id):
        return "❌ Dataset did not finish syncing in time."

    if not datasource_ids:
        return "❌ Data source creation failed."

    # Step 5: Create Session
    try:
        session_headers = {
            "x-pd-api-key": POWERDRILL_KEY,
            "Content-Type": "application/json"
        }
        session_payload = {
            "name": "Powerdrill Session",
            "user_id": POWERDRILL_USERID,
            "job_mode": "DATA_ANALYTICS"
        }
        session_res = requests.post(SESSION_URL, headers=session_headers, json=session_payload)
        #print("Session response:", session_res.status_code, session_res.text)
        session_res.raise_for_status()
        session_id = session_res.json().get("data", {}).get("id")
        if not session_id:
            return "❌ Could not create session. Aborting job."
    except Exception as e:
        return f"❌ Session creation failed: {e}"

    # Step 6: Create Job
    try:
        job_payload = {
            "session_id": session_id,
            "user_id": POWERDRILL_USERID,
            "stream": False,
            "question": "You are a csv-tsv-xlsx Specialist Agent. Give the task_description focus only on the relevant files for you!" + task_description,
            "dataset_id": dataset_id,
            "datasource_ids": datasource_ids,
            "output_language": "AUTO",
            "job_mode": "DATA_ANALYTICS"
        }
        job_headers = {
            "x-pd-api-key": POWERDRILL_KEY,
            "Content-Type": "application/json"
        }
        job_res = requests.post(JOB_URL, headers=job_headers, json=job_payload)
        #print("Job creation response:", job_res.status_code, job_res.text)
        job_res.raise_for_status()
        parsed = extract_answer_and_sources(job_res.json())
        return parsed
    except Exception as e:
        return f"❌ Job creation failed: {e}"

