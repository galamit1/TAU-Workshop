import openai

# Set up your OpenAI API key
openai.organization = "tau-71"
openai.api_key = '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'


# Define a function to send a message to ChatGPT and get a response
def chat_with_gpt(initial_instructions, prompt):
    completion = openai.Completion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": initial_instructions},
      {"role": "user", "content": prompt}
    ]
    )

    return completion.choices[0].message["content"]

initial_instructions = "I'd like you to rephrase the following message using polite, soft and nice language. Then rephrase it using harsh, mean and impolite language."

prompt = input("Please enter your prompt:" + '\n')
response = chat_with_gpt(initial_instructions, prompt)
print(response)