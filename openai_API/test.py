import openai

# Set up your OpenAI API key
openai.api_key = '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'

# Define a function to send a message to ChatGPT and get a response
def chat_with_gpt(prompt):
    response = openai.Completion.create(
        engine='text-davinci-003',  # Specify the ChatGPT model
        prompt=prompt,
        max_tokens=100,  # Control the length of the response
        temperature=0.7,  # Control the randomness of the response
        n=1,  # Generate a single response
        stop=None  # Can specify a custom stop sequence
    )
    
    return response.choices[0].text.strip()

# Example usage
prompt = """here is the contents of my current directory:
    drwxr-sr-x 30 jg yandexcs  4096 Aug 22 05:52 ana3
drwxr-sr-x  5 jg yandexcs  4096 Aug  8 03:43 anaconda3
-rw-r--r--  1 jg yandexcs   576 Aug 22 05:56 awesome.slurm
-rw-r--r--  1 jg yandexcs   595 Aug  8 22:43 default.slurm
-rw-r--r--  1 jg yandexcs  6761 Aug  8 23:00 fine_tune_native_pytorch.py
-rw-r--r--  1 jg yandexcs 62011 Aug  8 19:26 finetunepractice1.err
-rw-r--r--  1 jg yandexcs  8250 Aug  8 19:26 finetunepractice1.out
-rw-r--r--  1 jg yandexcs  5891 Aug 22 16:36 fine_tune_yelp_native_pytorch.err
-rw-r--r--  1 jg yandexcs     5 Aug 22 16:35 fine_tune_yelp_native_pytorch.out
-rw-r--r--  1 jg yandexcs  2769 Aug  8 18:07 finetuningpractice1.py
-rw-r--r--  1 jg yandexcs   585 Aug 13 21:46 from_saved_model.py
drwxr-sr-x  2 jg yandexcs  4096 Aug  8 22:33 pretrained_on_yelp
-rw-r--r--  1 jg yandexcs     0 Aug  9 19:35 slurm-49883.out
-rw-r--r--  1 jg yandexcs     4 Aug  9 19:38 slurm-49886.out
drwxr-sr-x  5 jg yandexcs  4096 Aug  8 22:57 TAU-Workshop
-rw-r--r--  1 jg yandexcs     0 Aug 22 05:56 test.err
-rw-r--r--  1 jg yandexcs     5 Aug 22 05:57 test.out
drwxr-sr-x  3 jg yandexcs  4096 Aug  8 03:28 tmp
-rw-r--r--  1 jg yandexcs   634 Aug 22 06:00 train_yelp.slurm
-rw-r--r--  1 jg yandexcs     0 Aug  9 19:36 yaakov.err
-rw-r--r--  1 jg yandexcs     4 Aug  8 22:41 yaakov.out
drwxr-sr-x 28 jg yandexcs  4096 Aug  8 04:20 yanaconda3
I'd like to move all the files from here in to the folder TAU-Workshop except for anaconda3, ana3, and yanaconda3. I'd also like to overwrite any existing files there with the same name. What's a command that can do that?"""
# prompt = input("Please enter your prompt")
response = chat_with_gpt(prompt)
print(response)
