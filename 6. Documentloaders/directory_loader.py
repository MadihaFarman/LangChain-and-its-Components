# helps load multiple documents from a directory using different file loaders based on file types
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader,TextLoader,CSVLoader


loader = DirectoryLoader(
    path="/home/madiha/Downloads",
    glob="*.pdf",
    loader_cls=PyPDFLoader,
    )

#docs = loader.load()
docs = loader.lazy_load()

# print(len(docs))
# print(len(docs[0].page_content))
# print(len(docs[0].metadata))
for doc in docs:
    print(doc.page_content)
    print(doc.metadata)


# lazy_load is used when u want to load large number of documents from a directory without loading all at once in memory, it doesnt load all docs rather it returns a generator that loads one doc at a time when iterated
