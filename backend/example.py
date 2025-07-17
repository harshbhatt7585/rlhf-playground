import logging
import os
import argparse
from dataclasses import dataclass, asdict
from typing import Tuple
from datasets import load_dataset
from trainer.ppo_trainer import PPOTrainerWrapper
from dataset.prompt_dataset import PromptDataset
from dotenv import load_dotenv
import wandb

# Load environment variables
load_dotenv()

if not os.environ.get("WANDB_API_KEY"):
    raise EnvironmentError("Set WANDB_API_KEY in environment variable")

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

    @staticmethod
    def from_args(args: argparse.Namespace) -> 'TrainingConfig':
        """Create TrainingConfig from argparse Namespace."""
        config_dict = {k: v for k, v in vars(args).items() if hasattr(TrainingConfig, k)}
        return TrainingConfig(**config_dict)


def load_environment() -> str:
    """Load environment variables and return Hugging Face token."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN environment variable not set")
        raise ValueError("HF_TOKEN is required")
    return hf_token


def load_and_prepare_datasets(
    config: TrainingConfig, tokenizer
) -> Tuple[PromptDataset, PromptDataset]:
    """Load and prepare train and evaluation datasets."""
    logger.info("Loading dataset: %s", config.dataset_name)
    dataset = load_dataset(config.dataset_name, config.dataset_subset)
    train_data = dataset["train"].select(range(config.train_samples))
    eval_data = dataset.get("validation", dataset.get("test")).select(range(config.eval_samples))

    logger.info("Preparing prompts...")
    def make_prompts(batch):
        return [
            f"Write an abstract of this article: {x['article'][:config.prompt_max_length]}"
            for x in batch
        ]

    train_prompts = make_prompts(train_data)
    eval_prompts = make_prompts(eval_data)

    train_dataset = PromptDataset(train_prompts, tokenizer)
    eval_dataset = PromptDataset(eval_prompts, tokenizer)
    return train_dataset, eval_dataset


def initialize_trainer(config: TrainingConfig, hf_token: str) -> PPOTrainerWrapper:
    """Initialize and return the PPO trainer."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO Training Script with configurable args")
    # Add args matching TrainingConfig fields
    parser.add_argument("--model_name", type=str,
                        default=TrainingConfig.model_name,
                        help="Model name or path for the policy model")
    parser.add_argument("--reward_model_name", type=str,
                        default=TrainingConfig.reward_model_name,
                        help="Model name or path for the reward model")
    parser.add_argument("--per_device_train_batch_size", type=int,
                        default=TrainingConfig.per_device_train_batch_size,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=TrainingConfig.gradient_accumulation_steps,
                        help="Gradient accumulation steps")
    parser.add_argument("--num_ppo_epochs", type=int,
                        default=TrainingConfig.num_ppo_epochs,
                        help="Number of PPO epochs")
    parser.add_argument("--num_mini_batches", type=int,
                        default=TrainingConfig.num_mini_batches,
                        help="Number of mini-batches per epoch")
    parser.add_argument("--response_length", type=int,
                        default=TrainingConfig.response_length,
                        help="Max response length")
    parser.add_argument("--total_episodes", type=int,
                        default=TrainingConfig.total_episodes,
                        help="Total episodes for training")
    parser.add_argument("--upload_to_hf", action="store_true" if TrainingConfig.upload_to_hf else "store_false",
                        help="Flag to upload model to Hugging Face repo")
    parser.add_argument("--repo_id", type=str,
                        default=TrainingConfig.repo_id,
                        help="Hugging Face repository ID to push model")
    parser.add_argument("--dataset_name", type=str,
                        default=TrainingConfig.dataset_name,
                        help="Dataset name for loading data")
    parser.add_argument("--dataset_subset", type=str,
                        default=TrainingConfig.dataset_subset,
                        help="Subset or config name for the dataset")
    parser.add_argument("--train_samples", type=int,
                        default=TrainingConfig.train_samples,
                        help="Number of training samples to select")
    parser.add_argument("--eval_samples", type=int,
                        default=TrainingConfig.eval_samples,
                        help="Number of evaluation samples to select")
    parser.add_argument("--prompt_max_length", type=int,
                        default=TrainingConfig.prompt_max_length,
                        help="Max tokens for prompt truncation")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainingConfig.from_args(args)
    hf_token = load_environment()
    trainer = initialize_trainer(config, hf_token)
    train_dataset, eval_dataset = load_and_prepare_datasets(config, trainer.tokenizer)

    logger.info("Starting training with config: %s", asdict(config))
    trainer.train(train_dataset, eval_dataset)
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()
