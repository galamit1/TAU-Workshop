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

small_train_dataset = tokenized_train_datasets.shuffle(seed=42)
small_eval_dataset = tokenized_test_datasets.shuffle(seed=42)

if torch.cuda.is_available():
    print("cuda")
    device = torch.device("cuda")
    print("device count = ", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)  # Wrap the model with DataParallel
else:
    device = torch.device("cpu")
    print("cpu")

model.to(device)

metric = evaluate.load("accuracy")

"""Call `compute` on `metric` to calculate the accuracy of your predictions. Before passing your predictions to `compute`, you need to convert the predictions to logits (remember all 🤗 Transformers models return logits):"""

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



"""### DataLoader

Create a `DataLoader` for your training and test datasets so you can iterate over batches of data:
"""


train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

"""Load your model with the number of expected labels:"""

model.eval()

model = AutoModelForSequenceClassification.from_saved("/home/yandex/MLW2023/jg/pretrained_sarcasm_on_bert_full")

# For single tests of the models accuracy. Returns an int
def yaakov_single_test(input_str):
    batch = tokenizer(input_str, padding="max_length", truncation=True, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    return predictions.item()

