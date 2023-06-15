
# from __future__ import annotations
# from langchain.prompts.prompt import PromptTemplate
# from langchain.chat_models.openai import ChatOpenAI
# from langchain.callbacks.stdout import StdOutCallbackHandler
# import os
# from dotenv import load_dotenv
# from typing import Any, Dict, List, Optional
# from pydantic import Extra
# from langchain.chat_models import ChatOpenAI
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.llms import OpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.base_language import BaseLanguageModel
# from langchain.callbacks.manager import (
#     AsyncCallbackManagerForChainRun,
#     CallbackManagerForChainRun,
# )
# from langchain.chains.base import Chain
# from langchain.prompts.base import BasePromptTemplate
# from datetime import datetime
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.llms import OpenAI
# from langchain.memory import VectorStoreRetrieverMemory
# from langchain.chains import ConversationChain
# from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatOpenAI
# os.environ["OPENAI_API_KEY"] = os.getenv(load_dotenv() and "OPEN_API_KEY")


# def ask_agent(query, vectorstore, verbose=True, model='gpt-4'):
#     # In actual usage, you would set `k` to be a higher value, but we use k=1 to show that
#     # the vector lookup still returns the semantically relevant information
#     retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
#     memory = VectorStoreRetrieverMemory(retriever=retriever)
#     llm = ChatOpenAI(model=model)
        
#     _DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

#     Relevant pieces of previous conversation:
#     {history}

#     (You do not need to use these pieces of information if not relevant)

#     Current conversation:
#     Human: {input}
#     AI:"""
#     PROMPT = PromptTemplate(
#         input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
#     )
#     conversation_with_summary = ConversationChain(
#         llm=llm,
#         prompt=PROMPT,
#         # We set a very low max_token_limit for the purposes of testing.
#         memory=memory,
#         verbose=verbose
#     )
#     result = conversation_with_summary.predict(input=query)
#     return result
