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

# ===================================================================================== Prepare the dataset =======================================================================================================
train_path = "/home/yandex/MLW2023/jg/TAU-Workshop/train_labeled.csv"
test_path = "/home/yandex/MLW2023/jg/TAU-Workshop/test_labeled.csv"
test_dataset = load_dataset("csv", data_files=test_path, sep=",")
train_dataset = load_dataset("csv", data_files=train_path, sep=",")
train_ds = train_dataset["train"]
test_ds = test_dataset["train"]


# === PreProcessing

# tokenize according to the pretrained model
# MODEL_PATH = "helinivan/english-sarcasm-detector"
MODEL_PATH = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


def tokenize_function(examples):
    return tokenizer(examples["tweets"], padding="max_length", truncation=True, return_tensors="pt") #pt for pytorch tensors


tokenized_test_datasets = test_ds.map(tokenize_function, batched=True)
tokenized_test_datasets = tokenized_test_datasets.remove_columns(["tweets"])
tokenized_test_datasets = tokenized_test_datasets.rename_column("class", "labels")
tokenized_test_datasets.set_format("torch")

tokenized_train_datasets = train_ds.map(tokenize_function, batched=True)
tokenized_train_datasets = tokenized_train_datasets.remove_columns(["tweets"])
tokenized_train_datasets = tokenized_train_datasets.rename_column("class", "labels")
tokenized_train_datasets.set_format("torch")

small_train_dataset = tokenized_train_datasets.shuffle(seed=42)
small_eval_dataset = tokenized_test_datasets.shuffle(seed=42)


# ???
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

# ???

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
print(outputs.logits.shape)


if torch.cuda.is_available():
    print("cuda")
    device = torch.device("cuda")
    print("device count = ", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)  # Wrap the model with DataParallel
else:
    device = torch.device("cpu")
    print("cpu")

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

model.to(device)

progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        #        input_ids = batch["input_ids"].to(device)
        #        attention_mask = batch["attention_mask"].to(device)
        #        labels = batch["labels"].to(device)
        #
        #        # Now you can create a new dictionary to pass to the model
        #        inputs = {
        #            "input_ids": input_ids,
        #            "attention_mask": attention_mask,
        #            "labels": labels,
        #        }
        #
        #        outputs = model(**inputs)
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss.mean()
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

#    for batch in train_dataloader:
#        loss = outputs.loss

model.module.save_pretrained("/home/yandex/MLW2023/jg/pretrained_sarcasm_on_bert")
print("saved")

# evaluate

metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()


# For single tests of the models accuracy. Returns an int
def yaakov_single_test(input_str):
    batch = tokenizer(input_str, padding="max_length", truncation=True, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    return predictions.item()
