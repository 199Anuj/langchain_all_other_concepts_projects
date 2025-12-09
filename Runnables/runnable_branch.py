from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableBranch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile", temperature = 1.5)

prompt1 = PromptTemplate(
    template="Generate text abot topic:  {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()


prompt2 = PromptTemplate(
    template="Summarize the given text belwo 200 words : {text}",
    input_variables=["text"]
)

text_gen_chain = prompt1 | model | parser


branch_chain = RunnableBranch(
    (
        lambda x: len(x.split()) > 200,
        prompt2 | model | parser
    ),
    RunnablePassthrough()
)


final_chain = text_gen_chain | branch_chain

result = final_chain.invoke({"topic": "Technology Advances in the 21st Century"})

print("result:",result)
print("Length of the result:", len(result.split()))