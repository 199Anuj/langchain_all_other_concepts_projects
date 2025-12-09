from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile", temperature = 1.5)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a joke about a topic : {topic}",
    input_variables=["topic"]
)       

joke_gen_chain = prompt1 | model | parser

def word_count(text):
    return len(text.split())

parallel_chain = RunnableParallel({

       "Joke": RunnablePassthrough(),
       "Word_Count": RunnableLambda(lambda x: len(x.split()))

})

final_chain = joke_gen_chain | parallel_chain

print(final_chain.invoke({"topic": "Indian Politics"}))

