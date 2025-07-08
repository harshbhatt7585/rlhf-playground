# trainers/ppo_trainer.py
import torch
import logging
from typing import List, Dict, Any, Optional, Union
from trl import PPOTrainer, PPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase, AutoModelForSequenceClassification
from transformers.image_processing_utils import BaseImageProcessor
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.processing_utils import ProcessorMixin
from datasets import Dataset
import numpy as np
from datetime import datetime
import os


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RewardModel:
    def __init__(self, model_name: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    
    def score(self, inputs: Union[List[str], List[Dict[str, Any]]]) -> np.ndarray:
        """
        Score the inputs using the reward model
        """
        if isinstance(inputs, list) and isinstance(inputs[0], str):
            inputs = [{"text": text} for text in inputs]
        
        # Tokenize inputs
        tokenized_inputs = self.model.tokenizer(
            [inp['text'] for inp in inputs],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Forward pass through the model
        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)
        
        # Extract scores (logits)
        scores = outputs.logits.squeeze().cpu().numpy()
        
        return scores

class PPOTrainerWrapper:
    """
    Wrapper class for PPO training with TinyLlama model
    """
    
    def __init__(self, 
                 model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 learning_rate: float = 1.41e-5,
                 per_device_train_batch_size: int = 4,
                 gradient_accumulation_steps: int = 1,
                 num_ppo_epochs: int = 4,
                 num_mini_batches: int = 4,
                 response_length: int = 128,
                 log_with: str = "tensorboard",
                 exp_name: str = "ppo-training"):
        
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        
        # Check device capabilities
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using GPU: {gpu_name}")
            
            # Check if GPU supports mixed precision
            try:
                test_tensor = torch.randn(1, device=self.device, dtype=torch.float16)
                supports_fp16 = True
                logger.info("GPU supports float16")
            except Exception:
                supports_fp16 = False
                logger.warning("GPU does not support float16, using float32")
        else:
            supports_fp16 = False
            logger.info(f"Using CPU: {self.device}")
        
        self.supports_fp16 = supports_fp16
        
        # Initialize PPO config with correct parameter names
        self.config = PPOConfig(
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_ppo_epochs=num_ppo_epochs,
            num_mini_batches=num_mini_batches,
            response_length=response_length,
            exp_name=exp_name,
            seed=42,
            # Disable mixed precision to avoid bf16 issues
            bf16=False,
            fp16=False,
            # Use more conservative settings
            kl_coef=0.05,
            cliprange=0.2,
            vf_coef=0.1,
            gamma=1.0,
            lam=0.95,
            num_train_epochs=1,
            save_strategy="steps",
            save_steps=100,
            logging_steps=10,
            output_dir="./ppo_outputs"
        )
    

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load the policy model
        if torch.cuda.is_available() and self.supports_fp16:
            try:
                self.policy_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                logger.info("Policy model loaded with float16 precision")
            except Exception as e:
                logger.warning(f"Failed to load with float16, falling back to float32: {e}")
                self.policy_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
        else:
            self.policy_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            if torch.cuda.is_available():
                self.policy_model = self.policy_model.to(self.device)
        
        self.reward_model = RewardModel(model_name)
        
        # Load reference model (separate copy for PPO)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.policy_model.dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        if torch.cuda.is_available() and not hasattr(self.ref_model, 'device'):
            self.ref_model = self.ref_model.to(self.device)
        
        # Load value model (can be the same architecture as policy model)
        self.value_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.policy_model.dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        if torch.cuda.is_available() and not hasattr(self.value_model, 'device'):
            self.value_model = self.value_model.to(self.device)
        
        # Create a simple reward model (using the same architecture)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

        if torch.cuda.is_available() and not hasattr(self.reward_model, 'device'):
            self.reward_model = self.reward_model.to(self.device)
        
        logger.info("All models loaded successfully")
    
    def create_dataset(self, prompts: List[str]) -> Dataset:
        """
        Create a dataset from prompts for PPO training
        """
        # Create dataset with proper format for PPO
        dataset_dict = {
            'input_ids': [],
            'attention_mask': []
        }
        
        for prompt in prompts:
            # Tokenize the prompt
            tokenized = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=256
            )
            
            dataset_dict['input_ids'].append(tokenized['input_ids'].squeeze().tolist())
            dataset_dict['attention_mask'].append(tokenized['attention_mask'].squeeze().tolist())
        
        return Dataset.from_dict(dataset_dict)
    
    def initialize_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """
        Initialize the PPO trainer with datasets
        """
        self.trainer = PPOTrainer(
            args=self.config,
            processing_class=self.tokenizer,
            model=self.policy_model,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            train_dataset=train_dataset,
            value_model=self.value_model,
            eval_dataset=eval_dataset
        )
        
        logger.info("PPO Trainer initialized successfully")
    
    def train(self, training_prompts: List[str], eval_prompts: Optional[List[str]] = None):
        """
        Main training method that uses the trainer's built-in training loop
        """
        logger.info(f"Starting PPO training with {len(training_prompts)} prompts")
        
        train_dataset = self.create_dataset(training_prompts)
        eval_dataset = self.create_dataset(eval_prompts) if eval_prompts else None
        
        self.initialize_trainer(train_dataset, eval_dataset)
        
        self.trainer.train()
        
        logger.info("Training completed!")
    
    def generate_response(self, prompt: str, max_length: int = 128) -> str:
        """
        Generate response for a given prompt using the trained policy model
        """
        try:
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.policy_model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove the prompt)
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response."
    
    def save_model(self, output_dir: str):
        """
        Save the trained model
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            self.trainer.save_model(output_dir)
            logger.info(f"Model saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")


def main():
    """
    Main function to run PPO training
    """
    # Initialize trainer
    trainer = PPOTrainerWrapper(
        per_device_train_batch_size=2,  # Reduced for memory efficiency
        gradient_accumulation_steps=2,
        num_ppo_epochs=2,
        num_mini_batches=2,
        response_length=64  # Reduced for memory efficiency
    )
    
    # Sample training prompts
    training_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "How does photosynthesis work?",
        "What are the benefits of renewable energy?",
        "Describe the process of making coffee.",
        "What is machine learning?",
        "How do you solve a Rubik's cube?",
        "What is the difference between weather and climate?",
        "Explain the concept of compound interest.",
        "What are the main causes of climate change?"
    ]
    
    # Sample evaluation prompts
    eval_prompts = [
        "What is artificial intelligence?",
        "How does gravity work?",
        "What is the water cycle?"
    ]
    
    try:
        # Run training
        trainer.train(
            training_prompts=training_prompts,
            eval_prompts=eval_prompts
        )
        
        # Save final model
        trainer.save_model("./final_ppo_model")
        
        # Test generation
        test_prompt = "What is the meaning of life?"
        response = trainer.generate_response(test_prompt)
        logger.info(f"Test generation - Prompt: {test_prompt}")
        logger.info(f"Test generation - Response: {response}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()