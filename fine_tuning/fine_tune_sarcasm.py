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



def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Argument Reader")

    # Add arguments with default values

    parser.add_argument("-m", "--model_path", help="Path of model to fine tune", default="bert-base-cased")
    parser.add_argument("-t", "--train_path", help="Path of training data as csv", default="./train_labeled.csv")
    parser.add_argument("-v", "--test_path", help="Path of test data as csv", default="./test_labeled.csv")
    parser.add_argument("-s", "--save_model_path", help="Path of test data as csv", default="./pretrained_sarcasm_on_bert")


    # Parse the command-line arguments
    args = parser.parse_args()
    train_path = args.train_path
    test_path = args.test_path
    model_path = args.model_path
    save_model_path = args.save_model_path

    test_dataset = load_dataset("csv", data_files=test_path, sep=",")
    train_dataset = load_dataset("csv", data_files=train_path, sep=",")
    train_ds = train_dataset["train"]
    test_ds = test_dataset["train"]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["tweets"], padding="max_length", truncation=True, return_tensors="pt")

    tokenized_test_datasets = test_ds.map(tokenize_function, batched=True)
    tokenized_test_datasets = tokenized_test_datasets.remove_columns(["tweets"])
    tokenized_test_datasets = tokenized_test_datasets.rename_column("class", "labels")
    tokenized_test_datasets.set_format("torch")

    tokenized_train_datasets = train_ds.map(tokenize_function, batched=True)
    tokenized_train_datasets = tokenized_train_datasets.remove_columns(["tweets"])
    tokenized_train_datasets = tokenized_train_datasets.rename_column("class", "labels")
    tokenized_train_datasets.set_format("torch")

    train_dataset = tokenized_train_datasets.shuffle(seed=42)
    eval_dataset = tokenized_test_datasets.shuffle(seed=42)

    metric = evaluate.load("accuracy")

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(eval_dataset, batch_size=8)

    """Load your model with the number of expected labels:"""
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

    if torch.cuda.is_available():
        print("cuda")
        device = torch.device("cuda")
        print("device count = ", torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)  # Wrap the model with DataParallel
    else:
        device = torch.device("cpu")
        print("cpu")

    """
    Optimizer and learning rate scheduler
    [`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) optimizer from PyTorch:
    """

    optimizer = AdamW(model.parameters(), lr=5e-5)

    """
    Create the default learning rate scheduler
    [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer):
    """

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    model.train()
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


    model.module.save_pretrained(save_model_path)
    print("saved")

 if __name__ == "__main__":
    main()


# For single tests of the models accuracy. Returns an int
def single_test(input_str):
    batch = tokenizer(input_str, padding="max_length", truncation=True, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    return predictions.item()
