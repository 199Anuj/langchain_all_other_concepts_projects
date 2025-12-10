from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv  
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

url = 'https://www.amazon.in/s?k=macbook+air+m1&adgrpid=57974791574&ext_vrnc=hi&hvadid=398125079225&hvdev=c&hvlocphy=9303798&hvnetw=g&hvqmt=b&hvrand=11355492962824142915&hvtargid=kwd-1016374080699&hydadcr=26949_2178297&mcid=83a079a322113015841254cbefa56d69&tag=googinhydr1-21&ref=pd_sl_3vn42hds2q_b' 

loader = WebBaseLoader(url)

doc = loader.load()


parser = StrOutputParser()

model = ChatGroq(model = "llama-3.3-70b-versatile", temperature = 1.5)

template = PromptTemplate(
    template="Answer the question - {ques}, from the following text : {text}",
    input_variables=["ques", "text"]
)   

chain = template | model | parser

result = chain.invoke({"ques": 'What are the ley features of the device mentioned in the webpage?', "text": doc[0].page_content})

print(result)