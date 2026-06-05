from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import GoogleGenerativeAI

loader = CSVLoader(file_path='gpa_table.csv')

docs = loader.load()

print(len(docs))  # number of rows in the csv
print(docs[0])  # first row in the csv