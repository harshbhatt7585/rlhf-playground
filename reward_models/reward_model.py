# reward_models/reward_model.py
from transformers import AutoModelForSequenceClassification

def train_reward_model(pairs):
    # pairs = [(prompt, chosen, rejected), ...]
    # Fine-tune a classification head model on preferences
    pass