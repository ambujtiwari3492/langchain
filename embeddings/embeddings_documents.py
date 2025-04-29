from langchain_ollama import OllamaEmbeddings

llm = OllamaEmbeddings(model="llama3.2:1b")
text = [
    "I love playing",
    "I am good at cricket",
    "I love coding",
    "I solve leetcode daily"
]
response = llm.embed_documents(text)
print(len(response))
print(response[0])