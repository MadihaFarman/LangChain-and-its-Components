from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
loader = TextLoader('Cricket.txt', encoding='utf-8')
docs = loader.load()
print("type of docs : ",type(docs))
print("len of docs : ",len(docs))
print("type of first doc : ",type(docs[0]))
print("first  doc",docs[0])
print("content of first doc",docs[0].page_content)

model = GoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
prompt = PromptTemplate(
    template="Summarize the following poem:\n{poem}",
    input_variables=['poem']
)
parser = StrOutputParser()
chain = prompt | model | parser
result = chain.invoke({'poem': docs[0].page_content})

print(result)