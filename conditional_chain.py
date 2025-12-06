from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()


class Feedback(BaseModel):
    sentiment: Literal["Positive","Negative"] = Field(description="Give the sentiment of the feedback as Positive or Negative")
    

parser2 = PydanticOutputParser(pydantic_object=Feedback)

model = ChatGroq(model = "llama-3.3-70b-versatile", temperature = 1.5) 

prompt1 = PromptTemplate(
    template="classiy the sentiment if the following feedback text into positive or negative: \n{feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)
print("prompt1:", prompt1)

parser = StrOutputParser()

classifier_chain = prompt1 | model | parser2

# result = classifier_chain.invoke({"text": "The product quality is excellent and delivery was prompt."})


prompt2 = PromptTemplate(
    template="write an appropriate response to this {feedback} feedbck",
    input_variables=["feedback"]
)  

prompt3 = PromptTemplate(
    template="write an appropriate response to this {feedback} feedback",
    input_variables=["feedback"]
)

print("prompt2:", prompt2)

branch_chain = RunnableBranch(

    (lambda x: x.sentiment == "Positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "Negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "No valid sentiment detected.")
)

chain = classifier_chain | branch_chain

print(chain.invoke({"feedback": "The product quality is ugly and delivery was not prompt."}))
chain.get_graph().print_ascii()


