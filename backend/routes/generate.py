from pydantic import BaseModel
from typing import List
from fastapi import APIRouter, HTTPException
from openai import AsyncAzureOpenAI
import os



AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")  # your chat deployment
AZURE_API_VERSION = "2023-05-15"


if not all([AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_DEPLOYMENT]):
    raise RuntimeError(
        "One of AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT or "
        "AZURE_OPENAI_DEPLOYMENT_NAME is not set in the environment"
    )


router = APIRouter(prefix='/generate')

openai_client = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_API_VERSION,
)

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

# ———————— Endpoint ————————

@router.post("/preferences", response_model=PreferenceGenRes)
async def generate_preferences(req: PreferenceGenReq):
    # 1) Build the prompt from seed examples
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
        # 2) Call Azure OpenAI
        response = await openai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You generate prompt-completion preference pairs."},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.7,
        )

        raw_output = response.choices[0].message.content

        # 3) Parse into structured examples
        generated: List[GeneratedExample] = []
        blocks = raw_output.strip().split("Prompt:")
        for block in blocks[1:]:
            try:
                prompt_part, rest = block.split("Chosen:", 1)
                chosen_part, rejected_part = rest.split("Rejected:", 1)
                generated.append(GeneratedExample(
                    prompt=prompt_part.strip(),
                    chosen=chosen_part.strip(),
                    rejected=rejected_part.strip()
                ))
            except Exception:
                # you could log the parse error here if needed
                continue
        print(generated)

        return PreferenceGenRes(generated=generated)

    except Exception as e:
        # surface errors as HTTP 500
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
