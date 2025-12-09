from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile", temperature = 1.5)


#template1 = PromptTemplate(
#    "write a detailed report on {topic}", 
#    input_variables=["topic"])
#
#template2 = PromptTemplate(
#    "write 5 line summary on text: {text}", 
#    input_variables=["text"])


# Prompt templates (fixed)
template1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="write a 5-line summary on the following text:\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

# prompt1 = template1.format(topic= "Blackhole")
# 
# result = model.invoke(prompt1)
# 
# prompt2 = template2.invoke({"text": result.content})
# 
# final_result = model.invoke(prompt2)
# 
# print(final_result.content)

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "Blackhole"})

print(result)
