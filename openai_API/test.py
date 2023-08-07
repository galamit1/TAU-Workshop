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
prompt = "What is the capital of France?"
#prompt = input("Please enter your prompt")
response = chat_with_gpt(prompt)
print(response)
