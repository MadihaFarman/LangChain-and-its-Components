from langchain_openai import ChatOpenAI #Import ChatOpenAI not OpenAI , OpenAi is LM model
from dotenv import load_dotenv

load_dotenv()

chatModel = ChatOpenAI(model = 'gpt-4' , temperature=0 , max_completion_tokens=10)
# lower temp value gives same output each time it is invoked . 
# Lower temp values give deterministic output while value near to 2 gives creative 
# output and gives unique output each time. 
# max_completion_tokens = tokens (words) in output

result = chatModel.invoke('What is the capital of Pakistan?')

print(result)

#print(result.content)   # for fetching content part only