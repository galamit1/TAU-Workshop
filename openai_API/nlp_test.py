import openai

# Set up your OpenAI API key
openai.organization = "tau-71"
openai.api_key = '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'


# Define a function to send a message to ChatGPT and get a response
def chat_with_gpt(initial_instructions, prompt):
    completion = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=[initial_instructions, prompt]
    )

    return completion.choices[0].message["content"]


# initial_instructions = "I'd like you to rephrase the following message using polite, soft and nice language. Then rephrase it using harsh, mean and impolite language."
#
# prompt = input("Please enter your prompt:" + '\n')
# response = chat_with_gpt(initial_instructions, prompt)
# print(response)


prompt_text = """What is the sentiment of the following tweet
tweet: I liked that the movie finished earlier. It was not worth watching.
sentiment: """

response = openai.Completion.create(
        model="text-davinci-003",
        prompt = prompt_text,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0,
    )

print(response)

completion = openai.ChatCompletion.create( # Change the function Completion to ChatCompletion
  model = 'gpt-3.5-turbo',
  messages = [ # Change the prompt parameter to the messages parameter
    {'role': 'user', 'content': 'Hello!'}
  ],
  temperature = 0
)

print(completion)
# message_type['choices'][0]['text']
