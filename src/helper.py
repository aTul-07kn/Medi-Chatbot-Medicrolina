from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings



def load_data(data):
    loader=DirectoryLoader(
        data, 
        glob="*.pdf", 
        loader_cls=PyPDFLoader,
        show_progress=True)

    docs=loader.load()
    return docs


def split_text(extracted_docs):
    splitter=RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=40)
    split_chunks=splitter.split_documents(extracted_docs)
    return split_chunks


def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings