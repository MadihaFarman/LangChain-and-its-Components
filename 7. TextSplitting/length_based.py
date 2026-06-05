from langchain_text_splitters import CharacterTextSplitter  # type: ignore[import]
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Pydantic_vs_TypedDict_Guide.pdf")
# text = "LangChain is a framework for developing applications powered by language models. It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more."

docs = loader.load()
splitter = CharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=0,
    separator="",
)

result = splitter.split_documents(docs)
print(result)