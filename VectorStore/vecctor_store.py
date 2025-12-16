from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from chromadb.config import Settings
from dotenv import load_dotenv
from uuid import uuid4
import os

load_dotenv()

embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")

docs = [
    Document(page_content="Virat Kohli is one of the most successful batsmen.", metadata={"team": "RCB"}),
    Document(page_content="Rohit Sharma led MI to five titles.", metadata={"team": "MI"}),
    Document(page_content="MS Dhoni is Captain Cool.", metadata={"team": "CSK"}),
]

texts = [d.page_content for d in docs]
metadatas = [d.metadata for d in docs]
ids = [str(uuid4()) for _ in texts]

print("Embedding documents...", flush=True)
embeddings = embedder.embed_documents(texts)
print("Embeddings done", flush=True)

vector_store = Chroma(
    collection_name="sample",
    persist_directory="my_chroma_db",
    embedding_function=None,
    client_settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

for i in range(len(texts)):
    vector_store._collection.add(
        documents=[texts[i]],
        metadatas=[metadatas[i]],
        embeddings=[embeddings[i]],
        ids=[ids[i]]
    )
    print(f"Inserted {i+1}", flush=True)

vector_store.persist()

print("✅ Files added to vector store", flush=True)
