from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

propmt = PromptTemplate(
    template = 'Give me 5 interesting facts about {topic}',
    input_variables= ['topic']
)

parser = StrOutputParser()

chain = propmt| model | parser

result = chain.invoke({'topic':'Cricket'})

print(result)