import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
# OPEN_API_KEY =os.getenv("OPEN_API_KEY")
# print(OPEN_API_KEY)
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="llama3.2:1b")
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""You are an experienced speech writer. 
    You need to craft an impactful title for a speech  
    on the following topic: {topic} 
    Answer exactly with one title. 
    """
)

speech_prompt = PromptTemplate(
    input_variables=["title","emotion"],
    template="""You need to write a powerful {emotion} speech of 350 words
     for the following title: {title}
     Format the output in json format with title and content as key and there respective value as value
     The format should be :
     {{
      "title: "Here Comes the title",
      "speech" :"the speech content"
      }}
    """
)

first_chain = title_prompt | llm | StrOutputParser() | (lambda title: (st.write(title),title)[1])
second_chain = speech_prompt | llm
final_chain = first_chain | (lambda title :{"title":title, "emotion":emotion}) |second_chain

st.title("Speech Generator")
topic = st.text_input("Enter the topic")
emotion = st.text_input("Enter the emotion")
if topic  and emotion:
    response =final_chain.invoke({"topic":topic})
    st.write(response.content)
