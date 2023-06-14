
from __future__ import annotations
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models.openai import ChatOpenAI
from langchain.callbacks.stdout import StdOutCallbackHandler
import os
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
from pydantic import Extra
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate
from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
os.environ["OPENAI_API_KEY"] = os.getenv(load_dotenv() and "OPEN_API_KEY")


def conversational_retrieval_chain(query, vectorstore):
    """Chat with the model."""
    qa = ConversationalRetrievalChain.from_llm(
        OpenAI(temperature=0, verbose=True), vectorstore.as_retriever(), return_source_documents=True)
    chat_history = []
    result = qa({"question": query, "chat_history": chat_history})
    print(result['source_documents'][0])
    return result
