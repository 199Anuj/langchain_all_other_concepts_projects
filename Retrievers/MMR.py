from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")

docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]


vectorestore = FAISS.from_documents(docs, embedder)

retriever = vectorestore.as_retriever(search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 1})


query = "What is langchain?"
results = retriever.invoke(query)


for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)