
from fastapi import APIRouter, HTTPException
import subprocess
from pydantic import BaseModel, Field
from uuid import uuid4
import json
import torch
import os
import subprocess
import uuid
from azure.ai.ml import MLClient
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment as MLEnvironment
from azure.identity import DefaultAzureCredential
import asyncio

from fastapi import WebSocket

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


class AzureTrainArgs(BaseModel):
    model_name: str = Field("HuggingFaceTB/SmolLM-135M-Instruct", description="Policy model name or path")
    reward_model_name: str = Field("HuggingFaceTB/SmolLM-135M-Instruct", description="Reward model name or path")
    per_device_train_batch_size: int = Field(1, ge=1)
    gradient_accumulation_steps: int = Field(1, ge=1)
    num_ppo_epochs: int = Field(1, ge=1)
    num_mini_batches: int = Field(1, ge=1)
    response_length: int = Field(10, ge=1)
    total_episodes: int = Field(1, ge=1)
    upload_to_hf: bool = Field(True)
    repo_id: str = Field("test-model")
    dataset_name: str = Field("ccdv/arxiv-summarization")
    dataset_subset: str = Field("section")
    train_samples: int = Field(10, ge=1)
    eval_samples: int = Field(10, ge=1)
    prompt_max_length: int = Field(2048, ge=1)


router  = APIRouter(prefix="/train")


### Global job tracker
JOB_STATUS = {}


@router.post("/ppo", response_model=PPOTrainRes)
async def ppo_train(req: PPOTrainReq):
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
async def train_reward_model(req: RewardModelReq):
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



@router.post("/ppo/train-azure-job", response_model=AzureTrainRes)
async def ppo_azure_submit_job(args: AzureTrainArgs):
    """
    Submit a CommandJob to Azure ML Compute using the Python SDK.
    All PPO training parameters are passed to the container's entrypoint.
    """
    try:
        # Validate environment
        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        workspace = os.getenv("AZURE_WORKSPACE_NAME")
        if not all([subscription_id, resource_group, workspace]):
            raise ValueError("Set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME env vars.")

        # Initialize ML client
        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace
        )

        # Build command string from args
        cmd_parts = ["python example.py"] + [f"--{k} {v}" for k, v in args.dict().items()]
        command_str = " ".join(cmd_parts)

        docker_env = MLEnvironment(
            name=f"ppo-docker-env-{uuid.uuid4()}",
            image="quntaacr.azurecr.io/ppo-train:test",
            description="Custom Docker environment for PPO training"
        )

        # Create and submit the job
        job_name = f"ppo-job-{uuid.uuid4()}"
        job = command(
            name=job_name,
            code="./",  # Context folder containing train.py
            command=command_str,
            environment=docker_env,
            compute="qunta-teslat4"
        )

        submitted = ml_client.jobs.create_or_update(job)
        return AzureTrainRes(job_id=submitted.name)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@router.websocket("/ws/logs/{job_id}")
async def websocket_logs(websocket: WebSocket, job_id: str):
    await websocket.accept()

    process = await asyncio.create_subprocess_exec(
        "az", "ml", "job", "stream", "--name", job_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    while True:
        line = await process.stdout.readline()
        if not line:
            break
        await websocket.send_text(line.decode())

    await websocket.close()





@router.get('/ppo/azure-status/{job_id}', response_model=AzureTrainStatus)
async def ppo_azure_status(job_id: str):
    try:
        result = subprocess.run(
            ['az', 'ml', 'job', 'show', '--name', job_id, '--query', 'status', '-o', 'tsv'],
            capture_output=True, text=True, check=True
        )
        status = result.stdout.strip()
        return AzureTrainStatus(job_id=job_id, status=status)

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status for job {job_id}: {e.stderr.strip()}")