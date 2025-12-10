from langchain_community.document_loaders.text import TextLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv  
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
loader = TextLoader("Cricket.txt", encoding="utf-8")

docs = loader.load()

print(f"Number of documents loaded: {len(docs)}")
print("Content of the document:")
print(type(docs[0]))        
print("Metadata of the document:")
print(docs[0].metadata)


parser = StrOutputParser()

model = ChatGroq(model = "llama-3.3-70b-versatile", temperature = 1.5)

template = PromptTemplate(
    template="Summarize the following POEM:\n{poem}",
    input_variables=["poem"]
)   

chain = template | model | parser

result = chain.invoke({"poem": docs[0].page_content})

print(result)