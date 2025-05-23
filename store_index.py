from src.helper import load_data
from src.helper import split_text
from src.helper import download_hugging_face_embeddings
import os
from dotenv import load_dotenv 
# Import the Pinecone library
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")

extracted_docs=load_data("Data/")
chunks=split_text(extracted_docs)
embeddings=download_hugging_face_embeddings()


# Initialize a Pinecone client with your API key
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create a dense index with integrated embedding
index_name = "medi-chatbot-index"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384, # Replace with your model dimensions
        metric="cosine", # Replace with your model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    
vectorstore_from_docs = PineconeVectorStore.from_documents(
        documents=chunks,
        index_name=index_name,
        embedding=embeddings
    )