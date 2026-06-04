from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser


# using two llms
load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a detailed report in {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from following text\n {text}",
    input_variables=['text']
)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic' : 'unemployment in Pakistan'})

print(result)
