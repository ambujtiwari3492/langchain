from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_ollama import ChatOllama
from langchain import hub
import streamlit as st
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools

llm=ChatOllama(model="llama3.2:1b")




prompt = hub.pull("hwchase17/react")

tools = load_tools(["wikipedia","ddg-search"])

agent= create_react_agent(llm,tools,prompt)

agent_executor=AgentExecutor(agent= agent,tools=tools,verbose=True)
st.title("AI Agent")
task=st.text_input("Assign me a task")

if task:
    response = agent_executor.invoke({"input":task})
    st.write(response["output"])














