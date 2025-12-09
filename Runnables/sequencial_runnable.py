from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile", temperature = 1.5)

template1 = PromptTemplate(
    template="Write a joke about a topic : {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()


template2 = PromptTemplate(
    template="Provide the expalnation of the joke : {joke}",
    input_variables=["joke"]
)

chain = RunnableSequence(template1, model, parser, template2, model, parser)


result = chain.invoke({"topic": "computers"})

print(result)