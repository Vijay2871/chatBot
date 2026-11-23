
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings


### Extract text from pdf 

def load_pdf(data):
    loader = DirectoryLoader(
        data, 
        glob="*.pdf",
         loader_cls=PyPDFLoader
         )
    documents = loader.load()
    return documents

def filter_docs(docs: List[Document]) -> List[Document]:
    """Keep only each document's page content and source metadata."""
    minimal_docs: List[Document] = []

    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src},
            )
        )

    return minimal_docs


## split docs into chunks

def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len
    )
    text_chunks = text_splitter.split_documents(minimal_docs)
    return text_chunks


def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return embeddings

