import os
from langchain_openai import ChatOpenAI

OPEN_API_KEY =os.getenv("OPEN_API_KEY")
print(OPEN_API_KEY)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPEN_API_KEY)
question =input("Enter te question")
response = llm.invoke(question)
print(response)