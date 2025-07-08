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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PPOTrainerWrapper:
    def __init__(self,
                 model_name: str = "microsoft/DialoGPT-small",
                 reward_model_name: str = "microsoft/DialoGPT-small",
                 learning_rate: float = 3e-6,
                 per_device_train_batch_size: int = 1,
                 gradient_accumulation_steps: int = 1,
                 num_ppo_epochs: int = 1,
                 num_mini_batches: int = 1,
                 response_length: int = 64,
                 total_episodes: int = 1000,
                 exp_name: str = "ppo-training-fixed"):

        self.model_name = model_name
        self.reward_model_name = reward_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

        self._load_models()

    def _load_models(self):
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.policy_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.reward_model_name,
            num_labels=1,
            trust_remote_code=True,
        )

        self.value_model = AutoModelForSequenceClassification.from_pretrained(
            self.reward_model_name,
            num_labels=1,
            trust_remote_code=True,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def prepare_dataset(self, dataset: Dataset, dataset_text_field: str = "prompt") -> Dataset:
        def tokenize(element):
            outputs = self.tokenizer(
                element[dataset_text_field],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_attention_mask=True,
            )
            return {
                "input_ids": outputs["input_ids"],
                "attention_mask": outputs["attention_mask"]
            }

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=self.config.dataset_num_proc,
        )

    def create_dataset_from_prompts(self, prompts: List[str]) -> Dataset:
        dataset = Dataset.from_dict({"prompt": prompts})
        return self.prepare_dataset(dataset)

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
        model_name="microsoft/DialoGPT-small",
        reward_model_name="microsoft/DialoGPT-small",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_ppo_epochs=1,
        num_mini_batches=1,
        response_length=32,
        total_episodes=100
    )

    logger.info("Loading dataset...")
    dataset = load_dataset("trl-internal-testing/descriptiveness-sentiment-trl-style", split="descriptiveness")

    eval_samples = 100
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))

    with PartialState().local_main_process_first():
        train_dataset = trainer.prepare_dataset(train_dataset, "prompt")
        eval_dataset = trainer.prepare_dataset(eval_dataset, "prompt")

    trainer.train(train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.save_model("./final_ppo_model_fixed")



if __name__ == "__main__":
    main()
