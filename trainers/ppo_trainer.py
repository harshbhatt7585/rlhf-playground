import torch
import logging
import os
import gc
from typing import List, Dict, Any, Optional
from trl import PPOTrainer, PPOConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from datasets import Dataset, load_dataset
from accelerate import PartialState
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from dataset.prompt_dataset import PromptDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class PPOTrainerWrapper:
    def __init__(self,
                 model_name: str = "microsoft/DialoGPT-small",
                 reward_model_name: str = "./reward_model",
                 learning_rate: float = 3e-6,
                 per_device_train_batch_size: int = 1,
                 gradient_accumulation_steps: int = 1,
                 num_ppo_epochs: int = 1,
                 num_mini_batches: int = 1,
                 response_length: int = 64,
                 total_episodes: int = 1000,
                 exp_name: str = "ppo-arxiv-abstracts"):

        self.model_name = model_name
        self.reward_model_name = reward_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        else:
            logger.info("Using CPU")

        self.config = PPOConfig(
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_ppo_epochs=num_ppo_epochs,
            num_mini_batches=num_mini_batches,
            response_length=response_length,
            exp_name=exp_name,
            total_episodes=total_episodes,
            seed=42,
            save_strategy="steps",
            save_steps=200,
            logging_steps=20,
            output_dir="./ppo_outputs",
            bf16=True,
        )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

        self._load_models()

    def _load_models(self):
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.policy_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.policy_model.gradient_checkpointing_enable()

        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.reward_model_name,
            num_labels=1,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

        self.value_model = AutoModelForSequenceClassification.from_pretrained(
            self.reward_model_name,
            num_labels=1,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
        

    def initialize_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        self.trainer = PPOTrainer(
            args=self.config,
            processing_class=self.tokenizer,
            model=self.policy_model,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            value_model=self.value_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        logger.info(f"Starting training with {len(train_dataset)} samples.")
        if torch.cuda.is_available():
            logger.info(f"Memory before: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        self.initialize_trainer(train_dataset, eval_dataset)

        self.trainer.train()
        logger.info("Training complete.")

    def save_model(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.trainer.save_model(output_dir)
        logger.info(f"Model saved to {output_dir}")

    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


def main():
    trainer = PPOTrainerWrapper(
        model_name="HuggingFaceTB/SmolLM-135M-Instruct",
        reward_model_name="HuggingFaceTB/SmolLM-135M-Instruct",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_ppo_epochs=1,
        num_mini_batches=1,
        response_length=512,
        total_episodes=100
    )

    logger.info("Loading arXiv summarization dataset...")
    train_dataset = load_dataset("ccdv/arxiv-summarization", "section")["train"].select(range(10))
    eval_dataset = load_dataset("ccdv/arxiv-summarization", "section")["validation"].select(range(10))


    # Truncate articles for both train and eval
    train_prompts = ["Write a abstract of this article:  "+ x["article"][:2048] for x in train_dataset]
    eval_prompts = ["Write a abstract of this article:  "+ x["article"][:2048] for x in eval_dataset]

    logger.info("Preparing PPO-compatible prompt dataset...")
    train_dataset = PromptDataset(train_prompts, trainer.tokenizer)
    eval_dataset = PromptDataset(eval_prompts, trainer.tokenizer)

    trainer.train(train_dataset=train_dataset, eval_dataset=eval_dataset)
    # trainer.save_model("./ppo_model_arxiv")


if __name__ == "__main__":
    main()
