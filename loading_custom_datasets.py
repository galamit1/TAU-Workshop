from datasets import load_dataset, DatasetDict, Dataset, ClassLabel, Value, Features, DatasetBuilder
from datasets.features import features

train_path = "Datasets/kaggle_Tweets_with_Sarcasm_and_Irony/train.csv"
# file_dict = {"train": [train_path]}  #adding later test with test path

class_names = ["Irony", "Sarcasm", "Regular", "Figurative"]
features = Features({'text': Value('string'), 'label': ClassLabel(names=class_names)})

ds = load_dataset("csv", data_files=train_path, sep=",")
ds
print(ds.column_names)
#

# csv_dataset = load_dataset("csv", data_files=train_path, sep=",", features=features)
# print(csv_dataset)

