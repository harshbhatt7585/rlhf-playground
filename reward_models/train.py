import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset as HFDataset, DatasetDict
import logging
import os
from typing import List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_preference_dataset() -> DatasetDict:
    data = {
        "train": [
            {
                "chosen": "The cat sat on the mat and purred happily.",
                "rejected": "The mat was sat by the cat, who meowed angrily."
            },
            {
                "chosen": "Water boils at 100°C at sea level.",
                "rejected": "Water freezes at 100°C at sea level."
            },
            {
                "chosen": "Python is a programming language.",
                "rejected": "Python is a type of snake only."
            },
            {
                "chosen": "The sun rises in the east.",
                "rejected": "The sun rises in the west."
            }
        ],
        "eval": [
            {
                "chosen": "The moon orbits the Earth.",
                "rejected": "The Earth orbits the moon."
            },
            {
                "chosen": "Fire is hot and can burn.",
                "rejected": "Fire is cold and refreshing."
            }
        ]
    }
    return DatasetDict({
        "train": HFDataset.from_list(data["train"]),
        "validation": HFDataset.from_list(data["eval"])
    })


class RewardDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        if hasattr(data, 'to_list'):
            self.samples = data.to_list()
        else:
            self.samples = list(data)
        self.tokenizer = tokenizer
        self.max_length = max_length

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chosen = self.samples[idx]["chosen"]
        rejected = self.samples[idx]["rejected"]

        chosen_enc = self.tokenizer(
            chosen,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        rejected_enc = self.tokenizer(
            rejected,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
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
            "attention_mask_rejected": torch.stack([f["attention_mask_rejected"] for f in features])
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
        # Return loss as a detached tensor for safe logging
        loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None


def train_reward_model(model_name="distilbert-base-uncased"):
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1
    )

    model.resize_token_embeddings(len(tokenizer))

    dataset = get_preference_dataset()
    train_dataset = RewardDataset(dataset["train"], tokenizer)
    eval_dataset = RewardDataset(dataset["validation"], tokenizer)
    data_collator = RewardDataCollator(tokenizer)

    os.makedirs("./reward_model", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

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
        report_to="none",
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


def test_reward_model(model_path="./reward_model", test_texts=None):
    if test_texts is None:
        test_texts = [
            "The cat sat on the mat and purred happily.",
            "The mat was sat by the cat, who meowed angrily.",
            "Water boils at 100°C at sea level.",
            "Water freezes at 100°C at sea level."
        ]

    logger.info("Loading trained reward model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    logger.info("Testing reward model...")
    with torch.no_grad():
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            reward = outputs.logits.squeeze().item()
            logger.info(f"Text: '{text}' -> Reward: {reward:.4f}")


if __name__ == "__main__":
    try:
        train_reward_model("distilbert-base-uncased")
        # Uncomment to test:
        # test_reward_model()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
