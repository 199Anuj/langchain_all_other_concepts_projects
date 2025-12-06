from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


prompt1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate( 
    template="write a 5-line summary of the following text:\n{text}",
    input_variables=["text"]
)

model = ChatGroq(model = "llama-3.3-70b-versatile", temperature = 1.5)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "Cricket"})

print(result)

print(chain.get_graph().print_ascii())
