from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile", temperature = 1.5)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name age city of a fictional person. \n {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# prompt = template.invoke({})
# 
# result= model.invoke(prompt)
# 
# final_result = parser.parse(result.content) 
# print(final_result)
# print(type(final_result))

chain = template | model | parser

print(chain.invoke({}))