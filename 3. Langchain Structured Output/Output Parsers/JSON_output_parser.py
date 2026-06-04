from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template = "Give me the name , age and city of a fictional person.\n{format_instruction}",
    input_variables = [],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

# prompt = template.format()

# result = model.invoke(prompt)
 
# #print(result)

# finalResult = parser.parse(result.content)
# print(finalResult)
# print(type(finalResult))

chain = template|model|parser
result = chain.invoke({})
print(result)

# drawback od JSON : we can't enforce schema like we do with StructuredOutputParser . For eg : we can't enforce the output to have Fact1 = age and fact2 = city etc.