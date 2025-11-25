import ollama
from langchain.prompts import ChatPromptTemplate
import os
import shutil

PROMPT_TEMPLATE = """
This is an example prompt template. You can input example data like below:
Data1: {data1}
Data2: {data2}

And so on and so forth.
"""
MODEL = 'llama3.1'

def inference():
    response = ollama.chat(
        model=MODEL,
        messages=[{'role': 'user', 'content': ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(notes=context, question=query)}],
        stream=True,
        )
    for chunk in response:
        print(chunk['message']['content'], end='', flush=True)

if __name__ == '__main__':
    inference()