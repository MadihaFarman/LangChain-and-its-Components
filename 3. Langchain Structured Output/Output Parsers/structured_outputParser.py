from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser,ResponseSchema
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

# let we want to generate 3 facts about a topic
# schema for guiding LLM about the output structure
schema = [
    ResponseSchema(name='fact_1',description='Fact 1 about the topic.'),
    ResponseSchema(name='fact_2',description='Fact 2 about the topic.'),
    ResponseSchema(name='fact_3',description='Fact 3 about the topic.'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 facts about the {topic}\n{format_instruction}', # the format instruction will be added by the parser
    input_variables = ['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# partial_variables are variables whose values are known at the time of template creation

# prompt = template.invoke({'topic': 'Black hole'})

# result = model.invoke(prompt)

# finalResult= parser.parse(result.content)

# print(finalResult)

chain = template | model | parser

result = chain.invoke({'topic': 'Black hole'})
print(result)


# drawback of structured output parser is that it can enforce scehma but cant do data validation like pydantic based structured output parser 