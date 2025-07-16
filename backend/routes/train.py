
from fastapi import APIRouter, HTTPException
import subprocess
from pydantic import BaseModel
from uuid import uuid4
import json
import torch
import os
import subprocess
import uuid

class PPOTrainReq(BaseModel):
    config: dict
    hf_dataset: str

class PPOTrainRes(BaseModel):
    job_id: str

class PPOTrainJob(BaseModel):
    job_id: str
    status: str
    completed: bool

class RewardModelReq(BaseModel):
    base_model_repo: str
    dataset: str

class RewardModelRes(BaseModel):
    job_id: str


class AzureTrainReq(BaseModel):
    job_file: str  # path to local YAML like 'job.yml'

class AzureTrainRes(BaseModel):
    job_id: str

class AzureTrainStatus(BaseModel):
    job_id: str
    status: str



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




@router.post("/reward_model", response_model=RewardModelRes)
def train_reward_model(req: RewardModelReq):
    job_id = str(uuid4())
    JOB_STATUS[job_id] = {"status": "starting", "completed": False}

    config = {
        "base_model_repo": req.base_model_repo,
        "dataset": req.dataset
    }
    config_path = f"/tmp/{job_id}_reward_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    try:
        use_gpu = torch.cuda.is_available() or torch.backends.mps.is_built()
        docker_cmd = [
            "docker", "run", "--rm",
            "--name", f"reward-job-{job_id}",
            "-v", f"{config_path}:/app/reward_config.json",
        ]
        # if use_gpu:
        #     docker_cmd += ["--gpus", "all"]

        docker_cmd += [
            "-e", f"WANDB_API_KEY={os.getenv('WANDB_API_KEY')}",
            "train_rm:test"
        ]

        subprocess.Popen(docker_cmd)
        JOB_STATUS[job_id]["status"] = "running"

    except Exception as e:
        JOB_STATUS[job_id]["status"] = "failed"
        JOB_STATUS[job_id]["completed"] = True
        raise HTTPException(status_code=500, detail=str(e))

    return RewardModelRes(job_id=job_id)


@router.get("/reward_model/status/{job_id}")
def reward_status(job_id: str):
    if job_id not in JOB_STATUS:
        raise HTTPException(status_code=404, detail="Job ID not found")
    job = JOB_STATUS[job_id]
    return PPOTrainJob(job_id=job_id, status=job["status"], completed=job["completed"])


@router.post('/ppo/train-azure-job', response_model=AzureTrainRes)
def ppo_azure_submit_job():
    try:
        result = subprocess.run(
            ['az', 'ml', 'job', 'create', '--file', '../infra/PPO/job.yml', '--query', 'name', '-o', 'tsv'],
            capture_output=True, text=True, check=True
        )
        job_id = result.stdout.strip()
        return AzureTrainRes(job_id=job_id)

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Azure ML job submission failed: {e.stderr.strip()}")



@router.get('/ppo/azure-status/{job_id}', response_model=AzureTrainStatus)
def ppo_azure_status(job_id: str):
    try:
        result = subprocess.run(
            ['az', 'ml', 'job', 'show', '--name', job_id, '--query', 'status', '-o', 'tsv'],
            capture_output=True, text=True, check=True
        )
        status = result.stdout.strip()
        return AzureTrainStatus(job_id=job_id, status=status)

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status for job {job_id}: {e.stderr.strip()}")