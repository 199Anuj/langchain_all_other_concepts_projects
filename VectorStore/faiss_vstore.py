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

print("Creating FAISS vector store...", flush=True)

# vector_store = FAISS.from_documents(docs, embedder)

index = faiss.IndexFlatL2(len(embedder.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embedder,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

vector_store.add_documents(documents=docs)
print("✅ Files added to FAISS vector store", flush=True)
print("Total docs:", vector_store.index.ntotal)


vector = vector_store.index.reconstruct(0)

print(type(vector))
print(len(vector))
print(vector[:10])


print(vector_store.docstore._dict)

print(vector_store.similarity_search_with_score("Who is bowler",k=2))

updated_virat_doc = Document(page_content="Virat Kohli has hit most number of odi centuries", metadata={"team": "RCB"}),

vector_store.update_document(document_id= "2aa94ee6-4cf6-4844-9218-c12bc7ad3625", document=updated_virat_doc)