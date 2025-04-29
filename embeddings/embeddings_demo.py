from langchain_ollama import OllamaEmbeddings
llm = OllamaEmbeddings(model="llama3.2:1b")
text = input("Enter the text ")
response = llm.embed_query(text)
print(response)