import os
# from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_community.chat_models import ChatOllama
# OPEN_API_KEY =os.getenv("OPEN_API_KEY")
# print(OPEN_API_KEY)

st.title("Ask anything")

llm = ChatOllama(model="llama3.2:1b")
question =st.text_input("Enter te question")
if question:
    response = llm.invoke(question)
    st.write(response.content)