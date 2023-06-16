"""Conversational Retrieval Chain."""
from __future__ import annotations
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
os.environ["OPENAI_API_KEY"] = os.getenv(load_dotenv() and "OPEN_API_KEY")


def conversational_retrieval_chain(query, vectorstore):
    """Chat with the model."""
    question = ConversationalRetrievalChain.from_llm(
        OpenAI(temperature=0, verbose=True),
        vectorstore.as_retriever(),
        return_source_documents=True
    )
    chat_history = []
    result = question({"question": query, "chat_history": chat_history})
    print(result['source_documents'][0])
    return result
