from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chatModel = ChatGoogleGenerativeAI(model = 'gemini-3-flash-preview' , temperature=0.9) # few tiers are free

result = chatModel.invoke("Write me a poem on tree")

print(result)