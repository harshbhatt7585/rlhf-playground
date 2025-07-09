import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset
from datasets import load_from_disk, Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import wandb  # <-- Added

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RewardDataset(TorchDataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.samples = list(data)
        self.tokenizer = tokenizer
        self.max_length = max_length

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        chosen_enc = self.tokenizer(
            chosen, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        rejected_enc = self.tokenizer(
            rejected, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )

        return {
            "input_ids_chosen": chosen_enc["input_ids"].squeeze(0),
            "attention_mask_chosen": chosen_enc["attention_mask"].squeeze(0),
            "input_ids_rejected": rejected_enc["input_ids"].squeeze(0),
            "attention_mask_rejected": rejected_enc["attention_mask"].squeeze(0)
        }


class RewardDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        return {
            "input_ids_chosen": torch.stack([f["input_ids_chosen"] for f in features]),
            "attention_mask_chosen": torch.stack([f["attention_mask_chosen"] for f in features]),
            "input_ids_rejected": torch.stack([f["input_ids_rejected"] for f in features]),
            "attention_mask_rejected": torch.stack([f["attention_mask_rejected"] for f in features]),
        }


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = model.device

        chosen_outputs = model(
            input_ids=inputs["input_ids_chosen"].to(device),
            attention_mask=inputs["attention_mask_chosen"].to(device)
        )
        rejected_outputs = model(
            input_ids=inputs["input_ids_rejected"].to(device),
            attention_mask=inputs["attention_mask_rejected"].to(device)
        )

        chosen_reward = chosen_outputs.logits.squeeze(-1)
        rejected_reward = rejected_outputs.logits.squeeze(-1)

        # Ensure dimensions
        if chosen_reward.dim() == 0:
            chosen_reward = chosen_reward.unsqueeze(0)
        if rejected_reward.dim() == 0:
            rejected_reward = rejected_reward.unsqueeze(0)

        loss = -torch.nn.functional.logsigmoid(chosen_reward - rejected_reward).mean()

        if return_outputs:
            return loss, {
                "chosen_reward": chosen_reward.detach(),
                "rejected_reward": rejected_reward.detach()
            }
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None


def train_reward_model(model_name="distilbert-base-uncased", dataset_path="dataset/arxiv_summarization_dataset"):
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1
    )

    model.resize_token_embeddings(len(tokenizer))

    logger.info("Loading dataset from disk...")
    full_dataset = load_from_disk(dataset_path)
    dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = RewardDataset(dataset["train"], tokenizer)
    eval_dataset = RewardDataset(dataset["test"], tokenizer)
    data_collator = RewardDataCollator(tokenizer)

    os.makedirs("./reward_model", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    wandb.init(project="reward-model", name="reward-training")  # <-- Added

    training_args = TrainingArguments(
        output_dir="./reward_model",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        logging_dir="./logs",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        logging_steps=10,
        remove_unused_columns=False,
        report_to="wandb",  # <-- Changed from "none"
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available()
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    logger.info("Starting training...")
    try:
        trainer.train()
        trainer.save_model("./reward_model")
        tokenizer.save_pretrained("./reward_model")
        logger.info("Training complete.")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        train_reward_model("distilbert-base-uncased", "dataset/arxiv_summarization_dataset")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
