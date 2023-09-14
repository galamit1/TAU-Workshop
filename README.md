# Detecting Sarcasm/Cynicism in Tweets

### Jacob Goldsmith, Gal Amit, Mika Hurvits
#### Tel Aviv University
##### September 2023
_____
This repository contains the code that has been done for the machine learning workshop at Tel Aviv University.

## Data

- **[Data Manipulation](https://github.com/galamit1/TAU-Workshop/tree/main/data_manipulation)**

  The code for creating the datasets that used for the baseline evaluation and the fine-tuning algorithms.

Certainly! Here's an updated section for the "Fine Tuning bert-base-cased" in your README to include instructions for running the fine-tuning script on any saved model and any data, as well as saving the fine-tuned model to any output:


## Fine Tuning bert-base-cased

- **[Fine Tuning](https://github.com/galamit1/TAU-Workshop/tree/main/fine_tuning)**

  This section provides the script for fine-tuning the `bert-base-cased` pretrained model to detect sarcasm/cynicism in tweets. You can use this script to fine-tune the model (or another model) on your own data and save it to a location of your choice.

### How to Fine-Tune the Model

1. Make sure you have Python and the required libraries installed. You can install the necessary libraries using `pip`:

   ```bash
   pip install transformers datasets torch
   ```

2. Prepare your data in CSV format. You should have separate CSV files for training and testing data.

3. Navigate to the `fine_tuning` directory:

   ```bash
   cd fine_tuning
   ```

4. Modify the script `fine_tune.py` as needed:

   - Update the command-line arguments for the model path (`-m`), training data path (`-t`), testing data path (`-v`), and save model path (`-s`) to specify your desired input data and model and save location.

5. Run the fine-tuning script by executing the following command:

   ```bash
   python fine_tune.py -m your_pretrained_model -t your_train_data.csv -v your_test_data.csv -s your_output_model_dir
   ```

   Replace `your_pretrained_model` with the path to the pretrained model you want to fine-tune, `your_train_data.csv` with the path to your training data, `your_test_data.csv` with the path to your testing data, and `your_output_model_dir` with the directory where you want to save the fine-tuned model.

6. The script will begin fine-tuning the model on your data. The progress will be displayed, and the fine-tuned model will be saved to the specified output directory upon completion.

7. You can now use this fine-tuned model to classify sarcasm/cynicism in tweets.

Feel free to adapt this script to your specific needs, such as changing hyperparameters or using a different pretrained model.


## Chat GPT Baseline Calculation

- **[Chat GPT Baseline](https://github.com/galamit1/TAU-Workshop/tree/main/openai/completion)**

  The `calculate_baseline.py` code loads the validation dataset and calculates the results by querying Chat GPT API for the classifications.

    For tokens optimizations and to avoid errors, there were done manipulations on the questions and the way we query the API by sending butches. 

- **[Chat GPT Chain Of Thoughts](https://github.com/galamit1/TAU-Workshop/tree/main/openai/completion)**

  The Chain Of Thoughts process is done automatically by enabling in the The `calculate_baseline.py` code:
    ```python
    CHAIN_OF_THOUGHTS = True
    ```
## Fine Tune OpenAI model

- **[Chat GPT Fine Tuning Ada Model](https://github.com/galamit1/TAU-Workshop/tree/main/openai/fine_tuning)**

  The Fine Tuning process flow is described in the code in `fine_tune_ada_model.py`, together with the expected output.
