# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.schema.runnable import RunnableParallel , RunnableBranch, RunnableLambda
# from langchain_core.output_parsers import PydanticOutputParser
# from pydantic import BaseModel , Field
# from typing import Literal

# load_dotenv()

# # Define the model
# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )

# model = ChatHuggingFace(llm=llm)

# parser = StrOutputParser()

# class Feedback(BaseModel):
#     sentiment : Literal['positive','negative'] = Field(description='Give the sentiment of the feedback')

# parser_2 = PydanticOutputParser(pydantic_object=Feedback)

# prompt_1 = PromptTemplate(
#     template="Classify the sentiment of the following feedback into positive or negative.\n {feedback}\n{format_instruction}",
#     input_variables=['feedback'],
#     partial_variables={'format_instruction': parser_2.get_format_instructions()}

# )

# classifer_chain = prompt_1 | model | parser_2

# prompt_2 = PromptTemplate(
#     template='Write an appropraite response to this positive feedback\n {feedback}',
#     input_variables=['feedback']
# )
# prompt_3 = PromptTemplate(
#     template='Write an appropraite response to this negative feedback\n {feedback}',
#     input_variables=['feedback']
# )

# branch_chain = RunnableBranch(
#     (lambda x: x.sentiment == 'positive', prompt_2 | llm | parser),
#     (lambda x: x.sentiment == 'negative', prompt_3 | llm | parser),
#     RunnableLambda(lambda x: "Couldn't determine sentiment.")
# )


# chain = classifer_chain | branch_chain

# result = chain.invoke({'feedback':'This is a terrible phone.'})

# print(result)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables  import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# ✅ Use correct task for chat model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)
#model = llm  # just use the raw HuggingFaceEndpoint


# Output parsers
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser_2 = PydanticOutputParser(pydantic_object=Feedback)

# Sentiment classification prompt
prompt_1 = PromptTemplate(
    template="Classify the sentiment of the following feedback into positive or negative.\nFeedback: {feedback}\n{format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser_2.get_format_instructions()}
)

# Chain to classify sentiment
# | is called the pipe operator
classifier_chain = prompt_1 | model | parser_2

# Positive and negative response prompts
prompt_2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback:\n{feedback}',
    input_variables=['feedback']
)
prompt_3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback:\n{feedback}',
    input_variables=['feedback']
)

# Branching logic
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt_2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt_3 | model | parser),
    RunnableLambda(lambda x: "Couldn't determine sentiment.")
)

# Final chain
chain = classifier_chain | branch_chain

# Test it
result = chain.invoke({'feedback': 'This is good phone.'})
print(result)