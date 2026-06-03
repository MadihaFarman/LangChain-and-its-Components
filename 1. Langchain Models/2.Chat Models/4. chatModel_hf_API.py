from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv


# Load environment variables (if you have your token in .env)
load_dotenv()

# Initialize the Hugging Face Endpoint
# The Connection (Base LLM)
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
    
)

# Wrap it into a ChatHuggingFace model
# The Interpreter (Chat Model Wrapper)
model = ChatHuggingFace(llm=llm)

# Ask a question
prompt = "### Instruction:\nWhat is the capital of Pakistan?\n### Response:"  #(### Instruction:, ### Response:). This is a common prompt template used to guide conversational models to provide a clean, direct answer.
result = model.invoke(prompt)

# Print the result
print(result.content)
