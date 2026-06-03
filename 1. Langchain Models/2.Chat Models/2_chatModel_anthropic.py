from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

chatModel = ChatAnthropic(model='claude-opus-4-20250514')

result = chatModel.invoke("What is the capital of Pakistan?")

print(result)