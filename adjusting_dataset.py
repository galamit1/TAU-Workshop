from datasets import Dataset
import pandas as pd
import Dataset

train_path = "Datasets/kaggle_Tweets_with_Sarcasm_and_Irony/train.csv"

df = pd.read_csv(train_path)
df = pd.DataFrame(df)
dataset = Dataset.from_pandas(df)utils.data.DataLoader(dataset, batch_size=32)

train_ds = Dataset.from_pandas(train_df, split="train")
test_ds = Dataset.from_pandas(test_df, split="test")