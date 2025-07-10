from transformers import AutoTokenizer
from typing import List
from datasets import Dataset


class PromptDataset(Dataset):

    def __init__(self, prompts: List[str], tokenizer: AutoTokenizer):
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        ) for prompt in prompts]
        print(prompts)
        self.prompts = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")["input_ids"]

        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"input_ids": self.prompts[idx]}