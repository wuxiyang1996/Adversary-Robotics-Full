import openai

openai.api_key = "Your OpenAI API Key"
def rephrase_attack(rephrasing_prefix, curr_prompt):
    response = openai.Completion.create(
                    engine="gpt-3.5-turbo-instruct",
                    prompt=rephrasing_prefix + curr_prompt,
                    temperature=0.99,
                    max_tokens=512,
                    n=1,
                    stop=".",
                )
    result = response.choices[0]["text"].replace("\n", "")
    return result