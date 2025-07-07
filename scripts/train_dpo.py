from trainers.dpo_trainer import DPOTrainer

def main():
    # Initialize the DPOTrainer
    dpo_trainer = DPOTrainer()

    # Start the training process
    dpo_trainer.train()

if __name__ == "__main__":
    main()