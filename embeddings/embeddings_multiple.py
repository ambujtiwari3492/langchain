from langchain_ollama import OllamaEmbeddings
import numpy as np

llm = OllamaEmbeddings(model="llama3.2:1b")
text1 = input("Enter the text ")
response1 = llm.embed_query(text1)
text2 = input("Enter another text ")
response2 = llm.embed_query(text2)
similarity_score = np.dot(response1,response2)
print(similarity_score)