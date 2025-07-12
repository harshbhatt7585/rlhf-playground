from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uuid import uuid4
import subprocess
import json
import uvicorn
import torch
from typing import List
import openai
import os

#  Azure OpenAI Setup
openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")  # Set this in env
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g. https://your-resource.openai.azure.com/
openai.api_version = "2023-05-15"
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")  # e.g. "gpt-4-azure"



app = FastAPI()

class PPOTrainReq(BaseModel):
    config: dict
    hf_dataset: str

class PPOTrainRes(BaseModel):
    job_id: str 

class PPOTrainJob(BaseModel):
    job_id: str
    status: str
    completed: bool


#
# Pydantic models
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

app = FastAPI()

# Store job status in-memory (for now)
JOB_STATUS = {}

@app.post('/train/ppo', response_model=PPOTrainRes)
def ppo_train(req: PPOTrainReq):
    job_id = str(uuid4())
    JOB_STATUS[job_id] = {"status": "starting", "completed": False}

    # Save config to a file that can be mounted into Docker
    config_path = f"/tmp/{job_id}_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "config": req.config,
            "hf_dataset": req.hf_dataset
        }, f)
    

    try:
        # Run Docker container in detached mode
        use_gpu = torch.cuda.is_available() or (
            torch.backends.mps.is_built()
        )
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


@app.get('/ppo/status/{job_id}', response_model=PPOTrainJob)
def check_status(job_id: str):
    if job_id not in JOB_STATUS:
        raise HTTPException(status_code=404, detail="Job ID not found")
    job = JOB_STATUS[job_id]
    return PPOTrainJob(job_id=job_id, status=job["status"], completed=job["completed"])





@app.post("/generate/preferences", response_model=PreferenceGenRes)
def generate_preferences(req: PreferenceGenReq):
    seed_text = "\n".join(
        f"Prompt: {ex.prompt}\nChosen: {ex.chosen}\nRejected: {ex.rejected}\n"
        for ex in req.seed_examples
    )

    prompt = f"""
You are a helpful assistant that generates training data for preference alignment (RLHF/DPO).
Given the format below, generate {req.num_generations} new preference examples.
Each example should include a prompt, a better (chosen) answer, and a worse (rejected) answer.

Examples:
{seed_text}

Now generate {req.num_generations} new examples in the same format:
"""

    try:
        response = openai.ChatCompletion.create(
            engine=AZURE_DEPLOYMENT,  # NOTE: Use deployment name, not model name
            messages=[
                {"role": "system", "content": "You generate prompt-completion preference pairs."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7
        )

        raw_output = response['choices'][0]['message']['content']

        # Parse output
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
                continue  # skip malformed

        return PreferenceGenRes(generated=generated)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", reload=True, port=8000)


