import os
import json
import subprocess
import torch
from uuid import uuid4
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# Azure OpenAI env vars
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_API_VERSION = "2023-05-15"

app = FastAPI()


### Models

class PPOTrainReq(BaseModel):
    config: dict
    hf_dataset: str

class PPOTrainRes(BaseModel):
    job_id: str

class PPOTrainJob(BaseModel):
    job_id: str
    status: str
    completed: bool

class PreferenceExample(BaseModel):
    prompt: str
    chosen: str
    rejected: str

class PreferenceGenReq(BaseModel):
    seed_examples: List[PreferenceExample]
    num_generations: int = 5

class GeneratedExample(BaseModel):
    prompt: str
    chosen: str
    rejected: str

class PreferenceGenRes(BaseModel):
    generated: List[GeneratedExample]

class RewardModelReq(BaseModel):
    base_model_repo: str
    dataset: str

class RewardModelRes(BaseModel):
    job_id: str


### Global job tracker
JOB_STATUS = {}


@app.on_event("startup")
async def startup_event():
    app.state.openai_client = AsyncAzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )


@app.post("/train/ppo", response_model=PPOTrainRes)
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


@app.get("/ppo/status/{job_id}", response_model=PPOTrainJob)
def check_status(job_id: str):
    if job_id not in JOB_STATUS:
        raise HTTPException(status_code=404, detail="Job ID not found")
    job = JOB_STATUS[job_id]
    return PPOTrainJob(job_id=job_id, status=job["status"], completed=job["completed"])


@app.post("/generate/preferences", response_model=PreferenceGenRes)
async def generate_preferences(req: PreferenceGenReq):
    seed_text = "\n".join(
        f"Prompt: {ex.prompt}\nChosen: {ex.chosen}\nRejected: {ex.rejected}\n"
        for ex in req.seed_examples
    )

    full_prompt = f"""
You are a helpful assistant that generates training data for preference alignment (RLHF/DPO).
Given the format below, generate {req.num_generations} new preference examples.
Each example should include a prompt, a better (chosen) answer, and a worse (rejected) answer.

Examples:
{seed_text}

Now generate {req.num_generations} new examples in the same format:
"""

    try:
        client: AsyncAzureOpenAI = app.state.openai_client

        response = await client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You generate prompt-completion preference pairs."},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.7
        )

        raw_output = response.choices[0].message.content

        # Parse
        generated = []
        blocks = raw_output.strip().split("Prompt:")
        for block in blocks[1:]:
            try:
                lines = block.strip().split("Chosen:")
                prompt_part = lines[0].strip()
                chosen_part, rejected_part = lines[1].strip().split("Rejected:")
                generated.append(GeneratedExample(
                    prompt=prompt_part.strip(),
                    chosen=chosen_part.strip(),
                    rejected=rejected_part.strip()
                ))
            except Exception:
                continue

        return PreferenceGenRes(generated=generated)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/train/reward_model", response_model=RewardModelRes)
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
        if use_gpu:
            docker_cmd += ["--gpus", "all"]

        docker_cmd += [
            "rlhf:test",  # Assuming same image; swap if needed
            "python", "train_reward_model.py", "--config=/app/reward_config.json"
        ]

        subprocess.Popen(docker_cmd)
        JOB_STATUS[job_id]["status"] = "running"

    except Exception as e:
        JOB_STATUS[job_id]["status"] = "failed"
        JOB_STATUS[job_id]["completed"] = True
        raise HTTPException(status_code=500, detail=str(e))

    return RewardModelRes(job_id=job_id)


@app.get("/reward_model/status/{job_id}")
def reward_status(job_id: str):
    if job_id not in JOB_STATUS:
        raise HTTPException(status_code=404, detail="Job ID not found")
    job = JOB_STATUS[job_id]
    return PPOTrainJob(job_id=job_id, status=job["status"], completed=job["completed"])


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
