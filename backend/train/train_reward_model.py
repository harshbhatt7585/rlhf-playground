import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset
from datasets import load_from_disk, Dataset as HFDataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import wandb 
from trainer.reward_trainer import RewardTrainer

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
            "attention_mask_rejected": rejected_enc["attention_mask"].squeeze(0),
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



    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Only run on eval set and if wandb is active
        if self.args.report_to and "wandb" in self.args.report_to:
            model = self.model.eval()
            device = model.device
            table = wandb.Table(columns=["index", "prompt (chosen)", "prompt (rejected)", "chosen_reward", "rejected_reward"])

            for i, batch in enumerate(self.get_eval_dataloader()):
                inputs = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    chosen_reward = model(
                        input_ids=inputs["input_ids_chosen"],
                        attention_mask=inputs["attention_mask_chosen"]
                    ).logits.squeeze(-1)

                    rejected_reward = model(
                        input_ids=inputs["input_ids_rejected"],
                        attention_mask=inputs["attention_mask_rejected"]
                    ).logits.squeeze(-1)

                    for j in range(chosen_reward.size(0)):
                        prompt_chosen = self.tokenizer.decode(
                            inputs["input_ids_chosen"][j],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True
                        )
                        prompt_rejected = self.tokenizer.decode(
                            inputs["input_ids_rejected"][j],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True
                        )

                        table.add_data(
                            i * chosen_reward.size(0) + j,
                            prompt_chosen,
                            prompt_rejected,
                            chosen_reward[j].item(),
                            rejected_reward[j].item()
                        )

            wandb.log({"eval_table/reward_comparison_table": table})

        return metrics


def train_reward_model(model_name="distilbert-base-uncased", dataset_path="dataset/arxiv_summarization_dataset", load_from_disk=True):
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

    if load_from_disk:
        full_dataset = load_from_disk(dataset_path)
        dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
    else:
        dataset = load_dataset(dataset_path)
    
    train_dataset = RewardDataset(dataset["train"], tokenizer)
    eval_dataset = RewardDataset(dataset["test"], tokenizer)
    data_collator = RewardDataCollator(tokenizer)

    os.makedirs("./reward_model", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    wandb.init(project="reward-model", name="reward-training")  # <-- Added

    training_args = TrainingArguments(
        output_dir="./reward_model",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        logging_dir="./logs",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        logging_steps=10,
        remove_unused_columns=False,
        report_to="wandb", 
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
        train_reward_model("microsoft/DialoGPT-small", "Anthropic/hh-rlhf", load_from_disk=False)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
