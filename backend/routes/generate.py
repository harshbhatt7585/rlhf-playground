import os
import json
import tempfile
from uuid import uuid4
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Load .env if needed
from dotenv import load_dotenv
load_dotenv()

router = APIRouter(prefix='/generate')

class PreferenceExample(BaseModel):
    prompt: str
    chosen: str
    rejected: str

class PreferenceGenReq(BaseModel):
    seed_examples: List[PreferenceExample]
    num_generations: int = 5
    upload_to_hf: bool = False
    hf_token: str | None = None
    repo_id: str | None = None

class GeneratedExample(BaseModel):
    prompt: str
    chosen: str
    rejected: str

class PreferenceGenRes(BaseModel):
    generated: List[GeneratedExample]
    upload_job_id: str | None = None
    upload_message: str | None = None

# Azure OpenAI config should be loaded externally and client passed or importable
from openai import AsyncAzureOpenAI

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_API_VERSION = "2023-05-15"

if not all([AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_DEPLOYMENT]):
    raise RuntimeError("Required Azure OpenAI env vars missing")

openai_client = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_API_VERSION,
)

@router.post("/preferences", response_model=PreferenceGenRes)
async def generate_preferences(req: PreferenceGenReq):
    # 1) Build prompt
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
        # Call Azure OpenAI
        resp = await openai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "Generate preference pairs."},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.7,
        )
        raw = resp.choices[0].message.content
        # Parse
        generated = []
        for block in raw.strip().split("Prompt:")[1:]:
            try:
                p, rest = block.split("Chosen:",1)
                c, r = rest.split("Rejected:",1)
                generated.append(GeneratedExample(
                    prompt=p.strip(), chosen=c.strip(), rejected=r.strip()
                ))
            except:
                continue

        upload_job_id = None
        upload_message = None
        
        # 2) Conditional upload using huggingface_hub
        if req.upload_to_hf:
            if not (req.hf_token and req.repo_id):
                raise HTTPException(status_code=400, detail="hf_token and repo_id required for upload_to_hf")
            
            try:
                # Initialize HF API
                api = HfApi(token=req.hf_token)
                job_id = str(uuid4())
                
                # Create temporary file with JSONL data
                with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp_file:
                    for ex in generated:
                        tmp_file.write(json.dumps(ex.dict()) + "\n")
                    tmp_path = tmp_file.name
                
                try:
                    # Check if repo exists, create if not
                    try:
                        api.repo_info(req.repo_id, repo_type="dataset")
                    except RepositoryNotFoundError:
                        create_repo(
                            repo_id=req.repo_id,
                            token=req.hf_token,
                            repo_type="dataset",
                            private=False
                        )
                    
                    # Upload file to the dataset repository
                    api.upload_file(
                        path_or_fileobj=tmp_path,
                        path_in_repo=f"preference_data_{job_id}.jsonl",
                        repo_id=req.repo_id,
                        repo_type="dataset",
                        token=req.hf_token
                    )
                    
                    upload_job_id = job_id
                    upload_message = f"Upload completed successfully. File: preference_data_{job_id}.jsonl"
                    
                finally:
                    # Clean up temporary file
                    try:
                        os.remove(tmp_path)
                    except:
                        pass
                        
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

        return PreferenceGenRes(
            generated=generated,
            upload_job_id=upload_job_id,
            upload_message=upload_message
        )
    except HTTPException as e:
        print(e)
        raise
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")