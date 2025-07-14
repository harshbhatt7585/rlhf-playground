
from fastapi import APIRouter, HTTPException
import subprocess
from pydantic import BaseModel
from uuid import uuid4
import json
import torch

class PPOTrainReq(BaseModel):
    config: dict
    hf_dataset: str

class PPOTrainRes(BaseModel):
    job_id: str

class PPOTrainJob(BaseModel):
    job_id: str
    status: str
    completed: bool


router  = APIRouter(prefix="/train")


### Global job tracker
JOB_STATUS = {}


@router.post("/ppo", response_model=PPOTrainRes)
def ppo_train(req: PPOTrainReq):
    job_id = str(uuid4())
    JOB_STATUS[job_id] = {"status": "starting", "completed": False}

    config_path = f"/tmp/{job_id}_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "config": req.config,
            "hf_dataset": req.hf_dataset
        }, f)

    try:
        use_gpu = torch.cuda.is_available() or torch.backends.mps.is_built()
        docker_cmd = [
            "docker", "run", "--rm",
            "--name", f"ppo-job-{job_id}",
            "-v", f"{config_path}:/app/config.json",
        ]

        if use_gpu:
            docker_cmd += ["--gpus", "all"]

        docker_cmd += [
            "rlhf:test",
            "python", "example.py", f"--config=/app/config.json"
        ]

        subprocess.Popen(docker_cmd)
        JOB_STATUS[job_id]["status"] = "running"

    except Exception as e:
        JOB_STATUS[job_id]["status"] = "failed"
        JOB_STATUS[job_id]["completed"] = True
        raise HTTPException(status_code=500, detail=str(e))

    return PPOTrainRes(job_id=job_id)


@router.get("/ppo/status/{job_id}", response_model=PPOTrainJob)
def check_status(job_id: str):
    if job_id not in JOB_STATUS:
        raise HTTPException(status_code=404, detail="Job ID not found")
    job = JOB_STATUS[job_id]
    return PPOTrainJob(job_id=job_id, status=job["status"], completed=job["completed"])
