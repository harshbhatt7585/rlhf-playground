from datasets import load_from_disk

dataset = load_from_disk("dataset/arxiv_summarization_dataset")
print(dataset[1]['prompt'])
