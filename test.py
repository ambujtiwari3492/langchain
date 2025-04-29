import os
# from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
OPEN_API_KEY =os.getenv("OPEN_API_KEY")
print(OPEN_API_KEY)

llm = ChatOllama(model="llama3.2:1b ")
question =input("Enter te question")
response = llm.invoke(question)
print(response)