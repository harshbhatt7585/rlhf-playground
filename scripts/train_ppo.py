from trainers.ppo_trainer import PPOTrainer

def main():
    # Initialize the PPOTrainer
    trainer = PPOTrainer()

    # Start the training process
    trainer.train()

if __name__ == "__main__":
    main()