# Import necessary libraries
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from transformers import Trainer
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
import evaluate
from tqdm.auto import tqdm
from torch.nn.parallel import DataParallel
import argparse

# Define the main function
def main():
    # Create an ArgumentParser object to handle command-line arguments
    parser = argparse.ArgumentParser(description="Argument Reader")

    # Add command-line arguments with default values
    parser.add_argument("-m", "--model_path", help="Path of model to fine-tune", default="bert-base-cased")
    parser.add_argument("-t", "--train_path", help="Path of training data as csv", default="./train_labeled.csv")
    parser.add_argument("-v", "--test_path", help="Path of test data as csv", default="./test_labeled.csv")
    parser.add_argument("-s", "--save_model_path", help="Path to save the fine-tuned model", default="./pretrained_sarcasm_on_bert")

    # Parse the command-line arguments
    args = parser.parse_args()
    train_path = args.train_path
    test_path = args.test_path
    model_path = args.model_path
    save_model_path = args.save_model_path

    # Load test and train datasets from CSV files
    test_dataset = load_dataset("csv", data_files=test_path, sep=",")
    train_dataset = load_dataset("csv", data_files=train_path, sep=",")
    train_ds = train_dataset["train"]
    test_ds = test_dataset["train"]

    # Initialize a tokenizer for BERT
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # Define a function to tokenize input examples
    def tokenize_function(examples):
        return tokenizer(examples["tweets"], padding="max_length", truncation=True, return_tensors="pt")

    # Tokenize the test dataset
    tokenized_test_datasets = test_ds.map(tokenize_function, batched=True)
    tokenized_test_datasets = tokenized_test_datasets.remove_columns(["tweets"])
    tokenized_test_datasets = tokenized_test_datasets.rename_column("class", "labels")
    tokenized_test_datasets.set_format("torch")

    # Tokenize the train dataset
    tokenized_train_datasets = train_ds.map(tokenize_function, batched=True)
    tokenized_train_datasets = tokenized_train_datasets.remove_columns(["tweets"])
    tokenized_train_datasets = tokenized_train_datasets.rename_column("class", "labels")
    tokenized_train_datasets.set_format("torch")

    # Shuffle the train and test datasets
    train_dataset = tokenized_train_datasets.shuffle(seed=42)
    eval_dataset = tokenized_test_datasets.shuffle(seed=42)

    # Load the evaluation metric (e.g., accuracy)
    metric = evaluate.load("accuracy")

    # Create data loaders for training and evaluation
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(eval_dataset, batch_size=8)

    # Load the model for sequence classification with the specified number of labels (2 in this case)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

    # Check if CUDA (GPU) is available and configure device
    if torch.cuda.is_available():
        print("cuda")
        device = torch.device("cuda")
        print("device count =", torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)  # Wrap the model with DataParallel for multiple GPUs
    else:
        device = torch.device("cpu")
        print("cpu")

    # Initialize the AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Create a learning rate scheduler
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Move the model to the selected device (CPU or GPU)
    model.to(device)

    # Create a progress bar for training
    progress_bar = tqdm(range(num_training_steps))

    # Set the model in training mode
    model.train()

    # Training loop
    for _ in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss.mean()
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    # Save the fine-tuned model
    model.module.save_pretrained(save_model_path)
    print("Model saved")

# Entry point for the script
if __name__ == "__main__":
    main()

# Function for single tests of the model's accuracy, returns an integer
def single_test(input_str):
    batch = tokenizer(input_str, padding="max_length", truncation=True, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    return predictions.item()
