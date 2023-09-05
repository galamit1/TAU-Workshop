import json

training_data = 'train_labeled.csv'
validation_data = 'test_labeled.csv'

training_file_name = "training_data.jsonl"
validation_file_name = "validation_data.jsonl"

MAX_SIZE = 100


def prepare_data(data_file, final_file_name):
    count = 0
    with open(data_file, "r") as data:
        with open(final_file_name, 'w') as outfile:
            data.readline()  # skip header
            for entry in data:
                count += 1
                if count >= MAX_SIZE:
                    break
                if len(entry) < 3:
                    continue
                classification = 'Sarcastic' if entry[-2] == "1" else 'Not Sarcastic'
                out = {"prompt": entry[:-2], "completion": " " + classification + "\n"}
                json.dump(out, outfile)
                outfile.write('\n')


prepare_data(training_data, training_file_name)
prepare_data(validation_data, validation_file_name)

###
# next, we will upload the files using these commands:
# openai tools fine_tunes.prepare_data -f "training_data.jsonl"
# openai tools fine_tunes.prepare_data -f "validation_data.jsonl"
# remember to save the file ID for the fine tuning call
# https://www.datacamp.com/tutorial/fine-tuning-gpt-3-using-the-open-ai-api-and-python
###
