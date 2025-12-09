from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile", temperature = 1.5)

schema = [
    ResponseSchema(name="Fact1", description="Fact1 about the topic"),
    ResponseSchema(name="Fact2", description="Fact2 about the topic"),
    ResponseSchema(name="Fact3", description="Fact3 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Provide 3 interesting facts about the topic: {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# prompt = template.invoke({"topic": "Blackholes"})   
# 
# result = model.invoke(prompt)
# 
# final_result = parser.parse(result.content)
# 
# print(final_result)

chain = template | model | parser

print(chain.invoke({"topic": "Blackholes"}))