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

from routes.generate import router as generate_router
from routes.train import router as train_rotuer
from routes.dataset import router as dataset_router

load_dotenv()

# Azure OpenAI env vars
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_API_VERSION = "2023-05-15"

app = FastAPI()

app.include_router(generate_router)
app.include_router(train_rotuer)
app.include_router(dataset_router)


@app.on_event("startup")
async def startup_event():
    app.state.openai_client = AsyncAzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
)






if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
