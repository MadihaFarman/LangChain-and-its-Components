# from langchain_core.prompts import ChatMessagePromptTemplate

# chat_tmeplate = ChatMessagePromptTemplate([
#        ('system', 'you are a helpful {domain} expert'),
#        ('human', 'Explain in simple terms, what is {topic}')
#        ])
# prompt = chat_tmeplate.invoke({'domain':'cricket', 'topic':'dusra'})
# print(prompt)

from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain} expert."),
    ("human", "Explain in simple terms, what is {topic}?")
])

# Fill in the variables
prompt = chat_template.format_messages(domain='cricket', topic='dusra')

# Print formatted messages
# for message in prompt:
#     print(message)

print(prompt)
