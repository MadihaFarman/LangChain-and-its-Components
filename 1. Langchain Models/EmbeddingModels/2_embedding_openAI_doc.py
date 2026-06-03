from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

docs = ['Islamabad is the capital of Pakistan' \
'Delhi is the capital of India ' \
'Beijing is the capital of China']

embedding = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=32)

result = embedding.embed_documents(docs)

print(str(result))