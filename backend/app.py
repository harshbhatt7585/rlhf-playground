import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from openai import AsyncAzureOpenAI
import uvicorn

from routes.generate import router as generate_router
from routes.train    import router as train_router 
from routes.dataset  import router as dataset_router

app = FastAPI()

app.include_router(generate_router)
app.include_router(train_router)
app.include_router(dataset_router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
