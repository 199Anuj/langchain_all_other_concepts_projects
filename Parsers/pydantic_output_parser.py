from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile", temperature = 1.5)


class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(gt=18, description="The age of the person")
    city: str = Field(description="The city where the person lives")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Give me the name age of a fictional person from {city} \n {format_instructions}",
    input_variables=["city"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# prompt = template.invoke({"city": "India"})
# 
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)

chain = template | model | parser
response = chain.invoke({"city": "Pakistan"})
print(response)