from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from chat_gpt import chat_with_gpt
import openai

openai.organization = "tau-71"
openai.api_key = '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'

TEST_PATH = "../train_labeled.csv"
INITIAL_INSTRUCTIONS = "I'd like you to classify the following tweets if it's a sarcasm or not, print 1 if you think it's sarcasm and 0 if not. the tweets split with \\n, write the results on after the other, for example the output can be 0100011"

TWEETS_IN_A_BUNCH = 5
NUMBER_OF_BUNCHES = 2


def get_predictions(ds):
    predictions = []
    for i in range(NUMBER_OF_BUNCHES):
        tweets_done = i * TWEETS_IN_A_BUNCH
        tweets_input = '\n'.join([str(tweet) for tweet in ds["tweets"][tweets_done: tweets_done + TWEETS_IN_A_BUNCH]])
        response = chat_with_gpt(INITIAL_INSTRUCTIONS, str(tweets_input))
        predictions += [int(res) for res in response]
    return predictions


def calculate_base_line(ds):
    y_pred = get_predictions(ds)
    y_true = [i.as_py() for i in ds["class"][:len(y_pred)]]

    # Calculate Precision
    precision = precision_score(y_true, y_pred)
    print("Precision:", precision)

    # Calculate Recall
    recall = recall_score(y_true, y_pred)
    print("Recall:", recall)

    # Calculate F1-Score
    f1 = f1_score(y_true, y_pred)
    print("F1-Score:", f1)

    # Calculate Support (this doesn't require scikit-learn)
    # It's simply the count of true instances for each class.
    support = [y_true.count(0), y_true.count(1)]
    print("Support for each class:", support)

    # Generate a classification report (includes precision, recall, and f1-score)
    class_report = classification_report(y_true, y_pred)
    print("Classification Report:\n", class_report)

    # Generate a confusion matrix
    confusion = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", confusion)


def main():
    ds = load_dataset("csv", data_files=TEST_PATH, sep=",")
    calculate_base_line(ds['train'].data)


if __name__ == '__main__':
    main()
