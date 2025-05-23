from flask import Flask, render_template, request 
from src.helper import *
from src.prompt import system_prompt
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv 
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
NVIDIA_API_KEY=os.getenv("NVIDIA_API_KEY")
os.environ["NVIDIA_API_KEY"]=NVIDIA_API_KEY

embeddings=download_hugging_face_embeddings()
index_name="medi-chatbot-index"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

llm = ChatNVIDIA(model="nvidia/llama-3.1-nemotron-70b-instruct", 
                 max_tokens=500, 
                 temperature=0.7, 
                 verbose=True)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

stuff_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, stuff_chain)

# @app.route('/')
# def hello():
#     return 'Hello, World!'

# @app.route("/help", methods=['POST'])
# def help():
#     return "helping myself"

# if __name__ == '__main__':
#     # Enables debug mode (auto–reload + better error messages)
#     app.run(port=8000, debug=True)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(port= 8080, debug= True)