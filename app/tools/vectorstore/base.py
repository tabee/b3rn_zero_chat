"""Module providingFunction for vectorstores."""
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

os.environ["OPENAI_API_KEY"] = os.getenv(load_dotenv() and "OPEN_API_KEY")
PATH_TO_WORKSPACE = os.getenv(load_dotenv() and "PATH_TO_WORKSPACE")
VECTORSTORE_DIRECTORY = PATH_TO_WORKSPACE+"/data/vectorstore"

embeddings = OpenAIEmbeddings()

def check_faiss_store_exists(
        pkl_path=VECTORSTORE_DIRECTORY+"/index.pkl",
        faiss_path=VECTORSTORE_DIRECTORY+"/index.faiss"
    ):
    """Function for checking if a faiss store exists."""
    store_exists = os.path.exists(pkl_path) and os.path.exists(faiss_path)
    if store_exists:
        print("FAISS-Store ist vorhanden.")
        return True
    else:
        print("FAISS-Store ist nicht vorhanden.")
        return False

def add_stuff_to_store(texts):
    """Function for creating a new vectorstore."""
    if check_faiss_store_exists():
        store = load_vectorstore()
        store.add_documents(documents=texts, embedding=embeddings)
        #store.from_documents(documents=texts, embedding=embeddings)
        store.save_local(VECTORSTORE_DIRECTORY)
    else:
        store = FAISS.from_documents(documents=texts, embedding=embeddings)
        store.save_local(VECTORSTORE_DIRECTORY)

def load_vectorstore():
    """Function for loading a local vectorstore."""
    store = FAISS.load_local(VECTORSTORE_DIRECTORY, embeddings=embeddings)
    return store

def get_embeddings():
    """Function for getting embeddings."""
    return embeddings
