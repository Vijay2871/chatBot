from dotenv import load_dotenv
import os
from src.helper import load_pdf, filter_docs, text_split, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY



extracted_docs = load_pdf(data="data/")
filtered_docs = filter_docs(extracted_docs)
text_chunks = text_split(filtered_docs)
embeddings = download_embeddings()


pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "cardbot1"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    index = pc.Index(index_name)

    docsearch = PineconeVectorStore.from_documents(
       documents=text_chunks,
        embedding=embeddings,
        index_name=index_name
    )