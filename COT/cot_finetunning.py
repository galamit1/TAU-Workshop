from datasets import load_dataset
from transformers import AutoTokenizer

train_path = "cynicism_reasoning_train.csv"
raw_data = load_dataset("csv", data_files=train_path, sep=',')
print(raw_data)

# checkpoint = "meta-llama/Llama-2-70b-chat-hf"
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)


