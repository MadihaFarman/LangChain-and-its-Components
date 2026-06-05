#pypdf loader is a doc loader that uses pypdf to read pdf , and used to load content from pdf files and convert each page into a document object
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('Pydantic_vs_TypedDict_Guide.pdf')

docs = loader.load()

print(docs)

print(len(docs))  # number of pages in the pdf

# pypdf loader doesnt work good with scanned pdfs as it cant extract text from images in scanned pdfs like having photos
# for them you can use : pdfplumber loader or ocr based loaders like textract loader , amazon textract loader etc,unstructured pdf loader ,unstructured ocr loader , pdfminer loader etc, which use ocr to extract text from images in scanned pdfs

# refer lanchain documentation for the implementation of other doc loaders :  https://docs.langchain.com/oss/python/langchain/overview