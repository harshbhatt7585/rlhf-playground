import logging
import os
from dataclasses import dataclass
from typing import List, Optional
from datasets import load_dataset
from trainer.ppo_trainer import PPOTrainerWrapper
from dataset.prompt_dataset import PromptDataset
from dotenv import load_dotenv
import wandb
import os

load_dotenv()

if not os.environ.get("WANDB_API_KEY"):
    raise EnvironmentError("Set WANDB_API_KEY in Environment Variable")

wandb.login(key=os.environ["WANDB_API_KEY"])


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct"
    reward_model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_ppo_epochs: int = 1
    num_mini_batches: int = 1
    response_length: int = 10
    total_episodes: int = 1
    upload_to_hf: bool = True
    repo_id: str = "test-model"
    dataset_name: str = "ccdv/arxiv-summarization"
    dataset_subset: str = "section"
    train_samples: int = 10
    eval_samples: int = 10
    prompt_max_length: int = 2048

def load_environment() -> str:
    """Load environment variables and return Hugging Face token."""
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN environment variable not set")
        raise ValueError("HF_TOKEN is required")
    return hf_token

def load_and_prepare_datasets(
    config: TrainingConfig, tokenizer
) -> tuple[PromptDataset, PromptDataset]:
    """Load and prepare train and evaluation datasets."""
    try:
        logger.info("Loading dataset: %s", config.dataset_name)
        dataset = load_dataset(config.dataset_name, config.dataset_subset)
        
        train_data = dataset["train"].select(range(config.train_samples))
        eval_data = dataset["validation"].select(range(config.eval_samples))

        logger.info("Preparing prompts...")
        train_prompts = [
            f"Write an abstract of this article: {x['article'][:config.prompt_max_length]}"
            for x in train_data
        ]
        eval_prompts = [
            f"Write an abstract of this article: {x['article'][:config.prompt_max_length]}"
            for x in eval_data
        ]

        train_dataset = PromptDataset(train_prompts, tokenizer)
        eval_dataset = PromptDataset(eval_prompts, tokenizer)
        return train_dataset, eval_dataset
    except Exception as e:
        logger.error("Failed to load or prepare datasets: %s", str(e))
        raise

def initialize_trainer(config: TrainingConfig, hf_token: str) -> PPOTrainerWrapper:
    """Initialize and return the PPO trainer."""
    try:
        logger.info("Initializing PPOTrainerWrapper...")
        return PPOTrainerWrapper(
            model_name=config.model_name,
            reward_model_name=config.reward_model_name,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_ppo_epochs=config.num_ppo_epochs,
            num_mini_batches=config.num_mini_batches,
            response_length=config.response_length,
            total_episodes=config.total_episodes,
            upload_to_hf=config.upload_to_hf,
            hf_token=hf_token,
            repo_id=config.repo_id
        )
    except Exception as e:
        logger.error("Failed to initialize trainer: %s", str(e))
        raise

def main():
    """Main function to orchestrate training process."""
    try:
        # Load configuration and environment
        config = TrainingConfig()
        hf_token = load_environment()

        # Initialize trainer
        trainer = initialize_trainer(config, hf_token)

        # Load and prepare datasets
        train_dataset, eval_dataset = load_and_prepare_datasets(config, trainer.tokenizer)

        # Start training
        logger.info("Starting training...")
        trainer.train(train_dataset, eval_dataset)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error("Training pipeline failed: %s", str(e))
        raise

if __name__ == "__main__":
    main()