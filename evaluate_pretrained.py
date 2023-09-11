import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datasets import load_dataset

# Load the fine-tuned model
MODEL_DIRECTORY = 'pretrained'
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIRECTORY)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Load your test dataset
TEST_PATH = "validation_labeled_2000_balanced.csv"
ds = load_dataset("csv", data_files=TEST_PATH, split="train", delimiter=",")

# Extract the input sentences and true labels
eval_sentences = ds["tweets"]
true_labels = ds["class"]

# Tokenize the evaluation sentences
input_ids = tokenizer(
    eval_sentences,
    truncation=True,
    padding=True,
    return_tensors="pt",
)

# Ensure input_ids is in the correct format (e.g., a PyTorch tensor)
with torch.no_grad():
    outputs = model(**input_ids)

# Extract the predicted class probabilities
logits = outputs.logits

# Calculate predicted labels
predicted_labels = torch.argmax(logits, dim=1).tolist()

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

# Calculate F1 score
f1 = f1_score(true_labels, predicted_labels, average="weighted")

# Generate a classification report
report = classification_report(true_labels, predicted_labels)

# Print the evaluation results
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Classification Report:\n", report)

# ouput for validation_labeled_2000_balanced:
# Accuracy: 0.891
# F1 Score: 0.8906692745555304
# Classification Report:
#                precision    recall  f1-score   support
#
#            0       0.94      0.84      0.88       500
#            1       0.85      0.95      0.90       500
#
#     accuracy                           0.89      1000
#    macro avg       0.90      0.89      0.89      1000
# weighted avg       0.90      0.89      0.89      1000