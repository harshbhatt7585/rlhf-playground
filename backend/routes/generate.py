from pydantic import BaseModel 
from typing import List
from fastapi import APIRouter, HTTPException
from openai import AsyncAzureOpenAI
import os

router = APIRouter("/generate")


AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_API_VERSION = "2023-05-15"


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





@router.post("/preferences", response_model=PreferenceGenRes)
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
        client: AsyncAzureOpenAI = router.state.openai_client

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
