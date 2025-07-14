from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from uuid import uuid4
import subprocess
import os

router = APIRouter(prefix="/dataset")


class UploadToHfReq(BaseModel):
    repo_id: str           # e.g., "username/my-dataset"
    dataset_path: str      # Local path to dataset folder
    hf_token: str          # Hugging Face token


class UploadToHfRes(BaseModel):
    job_id: str


@router.post("/upload_to_hf", response_model=UploadToHfRes)
def upload_to_huggingface(req: UploadToHfReq):
    job_id = str(uuid4())

    # Set HF token temporarily in the subprocess environment
    env = os.environ.copy()
    env["HF_TOKEN"] = req.hf_token

    try:
        subprocess.run([
            "huggingface-cli", "upload", req.dataset_path,
            "--repo-id", req.repo_id,
            "--type", "dataset",
            "--yes",
            "--token", req.hf_token
        ], check=True, env=env)

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    return UploadToHfRes(job_id=job_id)
