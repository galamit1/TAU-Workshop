import os

import numpy as np
import torch
from datasets import load_dataset, DatasetDict, Dataset, ClassLabel, Value, Features, DatasetBuilder
from transformers import AutoTokenizer, get_scheduler, BertTokenizerFast, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer
import evaluate
from torch.utils.data import DataLoader
from transformers import AdamW
# from tqdm.auto import tqdm
from torch.nn.parallel import DataParallel
from data_preparation import clean_tweet
import wandb

# import os
#
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

train_path = "train_labeled.csv"
validation_path = "validation_labeled.csv"  # the validation set is the same as the training set, need to change this
test_path = "test_labeled.csv"

test_dataset = load_dataset("csv", data_files=test_path, sep=",")
train_dataset = load_dataset("csv", data_files=train_path, sep=",")
train_ds = train_dataset["train"]
test_ds = test_dataset["train"]

raw_datasets = load_dataset("csv", data_files=file_dict, sep=",")
checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)


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
train_dataset = tokenized_train_datasets.shuffle(seed=42).select(range(20))
eval_dataset = tokenized_test_datasets.shuffle(seed=42).select(range(20))
# train_dataset = tokenized_datasets["train"]
# eval_dataset = tokenized_datasets["test"]


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # dynamic padding to speedup the training

# dataloaders converts our datasets into batches

train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=8, collate_fn=data_collator
    # originally should be the one in "validation" so if we differ test and validation should probably change it
)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

optimizer = AdamW(model.parameters(), lr=5e-5, no_deprecation_warning=True)

"""Lastly, specify `device` to use a GPU if you have access to one.
 Otherwise, training on a CPU may take several hours instead of a couple of minutes."""

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

print(num_training_steps)

if torch.cuda.is_available():
    print("cuda")
    device = torch.device("cuda")
    print("device count = ", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)  # Wrap the model with DataParallel
else:
    device = torch.device("cpu")
    print("cpu")


# TRAINING LOOP


def compute_metrics(eval_preds):
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    logits, labels = eval_preds
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# if we want to save our model to huggingfacehub
# from huggingface_hub import notebook_login
#
# notebook_login()

output_dir = "SarcasmModel-finetuned"
pretrained_dir = os.path.join(output_dir, 'pretrained')
training_args = TrainingArguments(output_dir=output_dir,
                                  overwrite_output_dir=True,
                                  evaluation_strategy="epoch",
                                  push_to_hub=False,
                                  report_to="wandb",
                                  # fp16=True, #for cuda only
                                  )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
# evel_results = trainer.evaluate()
# print(evel_results)
trainer.train()
wandb.finish()
# evel_results = trainer.evaluate()
# print(evel_results)
trainer.save_model(pretrained_dir)

# trainer.push_to_hub()


# ==================== training with accelerators (rather than the Trainer API) =====================

# metric = evaluate.load("accuracy")
# progress_bar = tqdm(range(num_training_steps))
# model.train()
# for epoch in range(num_epochs):
#     for batch in train_dataloader:
#         batch = {k: v.to(device) for k, v in batch.items()}
#         outputs = model(**batch)
#         loss = outputs.loss
#         loss.backward()
#
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         progress_bar.update(1)

# EVALUATION LOOP

# model.eval()
# for batch in eval_dataloader:
#     batch = {k: v.to(device) for k, v in batch.items()}
#     with torch.no_grad():
#         outputs = model(**batch)
#
#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
#     metric.add_batch(predictions=predictions, references=batch["labels"])
#
# print(metric.compute())
