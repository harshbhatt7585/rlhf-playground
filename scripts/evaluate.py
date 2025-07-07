# This script is used to evaluate the performance of the trained models. 
# It imports the necessary classes and functions to load models and compute evaluation metrics.

import sys
from models.load_model import load_model
from trainers.ppo_trainer import PPOTrainer
from trainers.dpo_trainer import DPOTrainer

def evaluate_model(model_path, trainer_type):
    model = load_model(model_path)
    
    if trainer_type == 'ppo':
        trainer = PPOTrainer(model)
    elif trainer_type == 'dpo':
        trainer = DPOTrainer(model)
    else:
        print("Invalid trainer type specified.")
        sys.exit(1)

    metrics = trainer.evaluate()
    print("Evaluation Metrics:", metrics)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <model_path> <trainer_type>")
        sys.exit(1)

    model_path = sys.argv[1]
    trainer_type = sys.argv[2]
    evaluate_model(model_path, trainer_type)