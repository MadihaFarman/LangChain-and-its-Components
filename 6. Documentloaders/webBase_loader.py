# used to load and extract content from web pages

# it uses BeautifulSoup under the hood to parse html content and extract text from it

# Limitations : may not handle very complex web pages with dynamic content loaded via javascript
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader

from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
prompt = PromptTemplate(
    template="Answer the following question \n{question} from the following text:\n{text}",
    input_variables=['question', 'text']
)
parser = StrOutputParser()
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Artificial_intelligence")
docs = loader.load()

# print(type(docs))
# print(len(docs)) 
# print(docs[0])
# print(docs[0].page_content)
# print(docs[0].metadata)

chain = prompt | model | parser

result = chain.invoke({'question': 'What is Artificial Intelligence?', 'text': docs[0].page_content})
print(result)
# also accpets list of urls to load multiple web pages at once