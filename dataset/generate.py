import os
import random
import logging
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
from transformers import pipeline
from rouge_score import rouge_scorer
from openai import AzureOpenAI


def load_environment():
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path)

    return {
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "deployment": "gpt-4.1-nano",
        "api_version": "2024-12-01-preview"
    }


def get_azure_openai_client(config):
    return AzureOpenAI(
        api_version=config["api_version"],
        azure_endpoint=config["endpoint"],
        api_key=config["api_key"],
    )


def generate_abstract(client, deployment, text):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who writes amazing abstracts of arXiv papers.",
            },
            {
                "role": "user",
                "content": f"Write a concise abstract for the following arXiv paper:\n\n{text}",
            }
        ],
        max_completion_tokens=800,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=deployment
    )
    return response.choices[0].message.content.strip()


def create_preference_dataset(dataset, client, deployment):
    preference_data = []

    for entry in dataset:
        article = entry["article"]
        abstract = entry["abstract"]
        new_abstract = generate_abstract(client, deployment, article)

        preference_data.append({
            "prompt": article,
            "chosen": abstract,
            "rejected": new_abstract
        })

    return Dataset.from_list(preference_data)


def save_dataset(dataset, save_path):
    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(save_path)
    print(f"Dataset saved to {save_path}")



def main():
    logging.basicConfig(level=logging.INFO)

    config = load_environment()
    client = get_azure_openai_client(config)

    # Load a sample of the dataset
    dataset = load_dataset("ccdv/arxiv-summarization", split="train[:100]")

    # Generate preference-based summaries
    preference_dataset = create_preference_dataset(dataset, client, config["deployment"])

    # Save the final dataset
    output_path = os.path.join(os.path.dirname(__file__), 'arxiv_summarization_dataset')
    save_dataset(preference_dataset, output_path)

if __name__ == "__main__":
    main()
