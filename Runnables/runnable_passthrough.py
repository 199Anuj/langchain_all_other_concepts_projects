from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile", temperature = 1.5)

prompt1 = PromptTemplate(
    template="Write a joke about a topic : {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()


prompt2 = PromptTemplate(
    template="Provide the expalnation of the joke : {joke}",
    input_variables=["joke"]
)

Joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    "Joke": RunnablePassthrough(),
    "Explanation": RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(Joke_gen_chain, parallel_chain)

final_result= final_chain.invoke({"topic": "Indian Politics"})

print(final_result)