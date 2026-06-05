# import os 
# os.environ["HUGGINGFACEHUB_ACCESS_TOKEN"] = ""
# Corrected Imports
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_chroma import Chroma                     
from langchain_core.documents import Document           
# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

from langchain_huggingface import HuggingFaceEndpointEmbeddings


embedding_model = embedding = HuggingFaceEndpointEmbeddings(
    model= 'sentence-transformers/all-MiniLM-L6-v2',
    task='feature-extraction'
)

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name='my_collection'
)
# Step 4: Convert vectorstore into a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
query = "What is Chroma used for?"
results = retriever.invoke(query)
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
results = vector_store.similarity_search(query, k=2)
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)


###