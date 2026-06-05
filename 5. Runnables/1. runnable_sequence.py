from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()


prompt1 = PromptTemplate(
    template='Write a joke about {topic}.',
    input_variables=['topic'],
)

prompt2 = PromptTemplate(
    template='Explain the following joke {text}.',
    input_variables=['text'],
)

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     task="text-generation"
# )

# model = ChatHuggingFace(llm=llm)
model = GoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)

parser = StrOutputParser()

chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)

print(chain.invoke({'topic':'AI'}))


# A RunnableSequence is the most fundamental composition object in LangChain. It is created automatically whenever you use the pipe operator (|) to chain two or more Runnables together.It acts as a container that handles the sequential execution of your pipeline, ensuring that the output of step $N$ is properly formatted and passed as the input to step $N+1$.