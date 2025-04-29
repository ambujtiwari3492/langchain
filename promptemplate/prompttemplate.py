import os
# from langchain_openai import ChatOpenAI
import streamlit as st
from langchain.chains.summarize.refine_prompts import prompt_template
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
# OPEN_API_KEY =os.getenv("OPEN_API_KEY")
# print(OPEN_API_KEY)

st.title("cuisine info")

llm = ChatOllama(model="llama3.2:1b")
prompt_template = PromptTemplate(
    input_variables=["country"],
    template="""You are an expert in traditional cuisines. 
    You provide information about a specific dish from a specific country. 
    Avoid giving information about fictional places. If the country is fictional 
    or non-existent answer: I don't know. 
    Answer the question: What is the traditional cuisine of {country}?
    """
)


country =st.text_input("Enter the country")
if country:
    response = llm.invoke(prompt_template.format(country=country))
    st.write(response.content)