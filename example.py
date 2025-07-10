import logging
from datasets import load_dataset
from trainers.ppo_trainer import PPOTrainerWrapper
from dataset.prompt_dataset import PromptDataset
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from dotenv import load_dotenv

load_dotenv()



def main():
    trainer = PPOTrainerWrapper(
        model_name="HuggingFaceTB/SmolLM-135M-Instruct",
        reward_model_name="HuggingFaceTB/SmolLM-135M-Instruct",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_ppo_epochs=1,
        num_mini_batches=1,
        response_length=10,
        total_episodes=1,
        upload_to_hf=True,
        hf_token=os.environ['HF_TOKEN'],
        repo_id="test-model"
    )
    print(os.environ['HF_TOKEN'])

    logger.info("Loading dataset...")
    raw_train = load_dataset("ccdv/arxiv-summarization", "section")["train"].select(range(10))
    raw_eval = load_dataset("ccdv/arxiv-summarization", "section")["validation"].select(range(10))

    train_prompts = [f"Write an abstract of this article: {x['article'][:2048]}" for x in raw_train]
    eval_prompts = [f"Write an abstract of this article: {x['article'][:2048]}" for x in raw_eval]

    train_dataset = PromptDataset(train_prompts, trainer.tokenizer)
    eval_dataset = PromptDataset(eval_prompts, trainer.tokenizer)

    trainer.train(train_dataset, eval_dataset)

if __name__ == "__main__":
    main()
