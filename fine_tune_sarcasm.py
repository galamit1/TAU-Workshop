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


model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

"""Lastly, specify `device` to use a GPU if you have access to one. Otherwise, training on a CPU may take several hours instead of a couple of minutes."""


if torch.cuda.is_available():
    print("cuda")
    device = torch.device("cuda")
    print("device count = ", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)  # Wrap the model with DataParallel
else:
    device = torch.device("cpu")
    print("cpu")

"""### Optimizer and learning rate scheduler

Create an optimizer and learning rate scheduler to fine-tune the model. Let's use the [`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) optimizer from PyTorch:
"""

optimizer = AdamW(model.parameters(), lr=5e-5)

"""Create the default learning rate scheduler from [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer):"""


num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

model.to(device)

"""<Tip>

Get free access to a cloud GPU if you don't have one with a hosted notebook like [Colaboratory](https://colab.research.google.com/) or [SageMaker StudioLab](https://studiolab.sagemaker.aws/).

</Tip>

Great, now you are ready to train! 🥳

### Training loop

To keep track of your training progress, use the [tqdm](https://tqdm.github.io/) library to add a progress bar over the number of training steps:
"""


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


"""### Evaluate

Just like how you added an evaluation function to [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer), you need to do the same when you write your own training loop. But instead of calculating and reporting the metric at the end of each epoch, this time you'll accumulate all the batches with `add_batch` and calculate the metric at the very end.
"""


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

"""<a id='additional-resources'></a>

## Additional resources

For more fine-tuning examples, refer to:

- [🤗 Transformers Examples](https://github.com/huggingface/transformers/tree/main/examples) includes scripts
  to train common NLP tasks in PyTorch and TensorFlow.

- [🤗 Transformers Notebooks](https://huggingface.co/docs/transformers/main/en/notebooks) contains various notebooks on how to fine-tune a model for specific tasks in PyTorch and TensorFlow.
"""
# For single tests of the models accuracy. Returns an int
def yaakov_single_test(input_str):
    batch = tokenizer(input_str, padding="max_length", truncation=True, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    return predictions.item()




