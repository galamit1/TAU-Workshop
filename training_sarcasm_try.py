import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer
import evaluate


train_path = "/home/yandex/MLW2023/mikahurvits/TAU-Workshop/train_labeled_5000_balanced.csv"
test_path = "/home/yandex/MLW2023/mikahurvits/TAU-Workshop/test_labeled_1000_balanced.csv"

test_dataset = load_dataset("csv", data_files=test_path, sep=",")
train_dataset = load_dataset("csv", data_files=train_path, sep=",")
train_ds = train_dataset["train"]
test_ds = test_dataset["train"]

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True, trust_remote_code=True)


# This function takes a dictionary (like the items of our dataset) and returns a new dictionary with the keys input_ids,
# attention_mask, and token_type_ids.
def tokenize_function(examples):
    return tokenizer(examples["tweets"], truncation=True)


tokenized_train_datasets = train_ds.map(tokenize_function, batched=True)
tokenized_test_datasets = test_ds.map(tokenize_function, batched=True)

tokenized_train_datasets = tokenized_train_datasets.remove_columns(["tweets"])
tokenized_train_datasets = tokenized_train_datasets.rename_column("class", "labels")
tokenized_train_datasets.set_format("torch")
tokenized_test_datasets = tokenized_test_datasets.remove_columns(["tweets"])
tokenized_test_datasets = tokenized_test_datasets.rename_column("class", "labels")
tokenized_test_datasets.set_format("torch")


# delete later and unindent two last rows. processing a small subset only tor testing
train_dataset = tokenized_train_datasets.shuffle(seed=42).select(range(100))
eval_dataset = tokenized_test_datasets.shuffle(seed=42).select(range(100))
# train_dataset = tokenized_datasets["train"]
# eval_dataset = tokenized_datasets["test"]


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # dynamic padding to speedup the training


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)



# TRAINING LOOP


def compute_metrics(eval_preds):
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


output_dir = "/home/yandex/MLW2023/mikahurvits/TAU-Workshop/SarcasmModel-finetuned"
training_args = TrainingArguments(output_dir=output_dir,
                                  overwrite_output_dir=True,
                                  num_train_epochs=60,
                                  per_device_train_batch_size=4,
                                  save_total_limit=2,
                                  save_strategy="epoch",
                                  report_to=["wandb"]
                                  )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)


trainer.train()
# evel_results = trainer.evaluate()
# print(evel_results)
# trainer.save_model(output_dir)
# wandb.finish()


