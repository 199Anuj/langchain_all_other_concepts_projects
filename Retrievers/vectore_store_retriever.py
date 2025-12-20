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
    Document(page_content="Virat Kohli is one of the most successful batsmen.", metadata={"team": "RCB"}),
    Document(page_content="Rohit Sharma led MI to five titles.", metadata={"team": "MI"}),
    Document(page_content="MS Dhoni is Captain Cool.", metadata={"team": "CSK"}),
    Document(page_content="Jusprit Bumrah is india's bowler who fastest 100 wicket takers in Tests", metadata={"team": "Mumbai Indians"}),

]

vectorestore = FAISS.from_documents(docs, embedder)

retriever = vectorestore.as_retriever(search_kwargs={"k": 2})

query = "Who is bowler"

result = retriever.invoke(query)

for i, doc in enumerate(result):
    print(f"Result {i+1}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")