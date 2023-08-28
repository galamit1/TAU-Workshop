import openai

# Set up your OpenAI API key
openai.organization = "tau-71"
openai.api_key = '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'

# my key
# openai.organization = "TAU"
# openai.api_key = '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'


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



