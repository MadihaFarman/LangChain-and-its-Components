from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
chat_history = []

llm = HuggingFaceEndpoint(
   repo_id="baidu/ERNIE-4.5-21B-A3B-PT",
   task="text-generation"
)

model = ChatHuggingFace(llm=llm)

while True:
    user_input = input('You: ')
    chat_history.append(user_input)
    if user_input=="exit":
        break
    result=model.invoke(chat_history)
    chat_history.append(result.content)
    print('AI: ', result.content)
print(chat_history)