import json
from example_data import training_data, validation_data

training_file_name = "training_data.jsonl"
validation_file_name = "validation_data.jsonl"


def prepare_data(dictionary_data, final_file_name):
    with open(final_file_name, 'w') as outfile:
        for entry in dictionary_data:
            json.dump(entry, outfile)
            outfile.write('\n')


prepare_data(training_data, "training_data.jsonl")
prepare_data(validation_data, "validation_data.jsonl")