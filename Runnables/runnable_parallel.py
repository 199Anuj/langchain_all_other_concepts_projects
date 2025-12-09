from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile", temperature = 1.5)

prompt1 = PromptTemplate(
    template="Generate a tweet about a topic : {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a LinkedIn Post about a topic : {topic}",    
    input_variables=["topic"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
"Tweet": RunnableSequence(prompt1, model, parser),
"LinkedIn Post": RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({"topic": "AI in healthcare"})

print("Tweet :::",result['Tweet'])
print("LinkediN ::: ",result['LinkedIn Post'])