import os
import numpy as np
import torch
from datasets import load_dataset, DatasetDict, Dataset, ClassLabel, Value, Features, DatasetBuilder
from transformers import AutoTokenizer, get_scheduler, BertTokenizerFast
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
import evaluate
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm.auto import tqdm
from torch.nn.parallel import DataParallel
import wandb


def train():
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)


    train_path = "/home/yandex/MLW2023/jg/TAU-Workshop/train_labeled.csv"
    test_path = "/home/yandex/MLW2023/jg/TAU-Workshop/test_labeled.csv"

    test_dataset = load_dataset("csv", data_files=test_path, sep=",")
    train_dataset = load_dataset("csv", data_files=train_path, sep=",")
    train_ds = train_dataset["train"]
    test_ds = test_dataset["train"]

    """As you now know, you need a tokenizer to process the text and include a padding and truncation strategy to handle any variable sequence lengths. To process your dataset in one step, use 🤗 Datasets [`map`](https://huggingface.co/docs/datasets/process.html#map) method to apply a preprocessing function over the entire dataset:"""

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


    def tokenize_function(examples):
        return tokenizer(examples["tweets"], padding="max_length", truncation=True, return_tensors="pt")


    tokenized_test_datasets = test_ds.map(tokenize_function, batched=True)
    tokenized_train_datasets = train_ds.map(tokenize_function, batched=True)

    tokenized_test_datasets = tokenized_test_datasets.remove_columns(["tweets"])
    tokenized_test_datasets = tokenized_test_datasets.rename_column("class", "labels")
    tokenized_test_datasets.set_format("torch")

    tokenized_train_datasets = tokenized_train_datasets.remove_columns(["tweets"])
    tokenized_train_datasets = tokenized_train_datasets.rename_column("class", "labels")
    tokenized_train_datasets.set_format("torch")

    """If you like, you can create a smaller subset of the full dataset to fine-tune on to reduce the time it takes:"""

    train_dataset = tokenized_train_datasets.shuffle(seed=42)
    eval_dataset = tokenized_test_datasets.shuffle(seed=42)

def compute_metrics(eval_preds):
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

'''
# 1. Start a W&B Run
wandb.init(entity='', project='eval_from_saved_pretrained_sarcasm_full')

# 2. Save mode inputs and hyperparameters
wandb.config.learning_rate = 0.01
'''


def trainer():
    # Model training code goes here
    output_dir = "Sarcasm_from_saved_finetuned_eval_results"
    pretrained_dir = os.path.join(output_dir, 'pretrained')
    training_args = TrainingArguments(output_dir=output_dir,
                                      overwrite_output_dir=True,
                                      evaluation_strategy="epoch",
                                      push_to_hub=False,
                                      fp16=True, #for cuda only
                                      report_to="wandb"
                                      )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     tokenizer=tokenizer,
    #     compute_metrics=compute_metrics,
    # )
    # trainer.evaluate()
    #
    # print(evel_results)
'''
# 3. Log metrics over time to visualize performance
wandb.log({"loss": loss})

# 4. Log an artifact to W&B
wandb.log_artifact(model)
'''

def main():
    TEST_PATH = 'test_labeled_1000_balanced.csv'

    # Import model and data
    model = AutoModelForSequenceClassification.from_pretrained("pretrained", num_labels=2)
    print("model loaded from pretrained!")

    test_dataset = load_dataset("csv", data_files=TEST_PATH, sep=",")
    dataloader = test_dataset


if __name__ == '__main__':
    main()