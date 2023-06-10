"""Python file to serve as the frontend"""
import os
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from ai import chain_01
from tools import vectorstore
os.environ["OPENAI_API_KEY"] = os.getenv(load_dotenv() and "OPEN_API_KEY")
embeddings = vectorstore.get_embeddings()
store = vectorstore.load_vectorstore()

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    template = """
    Antworte auf die Frage am Ende. Wenn Du keine zitierbare Antwort im Kontexttext findest, dann sage:
    ich weiss es nicht. Analysiere folgenden Kontexttext und versuche damit die folgende Frage 
    zu beantworten. Versuche aus dem Kontexttext zu zitieren. \n\n
    {contexttext} \n\n 
    Frage: {question}\n  
    Antwort:"
    Wichtig: Du musst eine Antowrt geben, die im Kontexttext zitiert werden kann und die Quelle/Source IMMER nennen.
    Regel: Wenn keine Quelle/Source im Kontexttext vorhanden die zur Antwort passt, dann sage: ich weiss es nicht.
    """
    prompt = PromptTemplate(template=template, input_variables=["contexttext", "question"])
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0, max_tokens=3000), verbose=True)
    return llm_chain

chain = load_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="AI Char Demo für Kasorn Asaaipa", page_icon=":robot:")
st.header("QA Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "was finde ich auf der website ch.ch?", key="input")
    return input_text

user_input = get_text()

if user_input:
    #output = chain.run(input=user_input)
    embeddings = embeddings
    retriever = store.as_retriever()
    compressed_docs_transformer = chain_01.contextual_compression_document_transformer(user_input, embeddings, retriever)
    contenttext = ""
    for i in range(3):
        contenttext += "\n\n" + compressed_docs_transformer[i].page_content +"\n" + compressed_docs_transformer[i].metadata["source"]+"\n\n"
    output = chain.predict(contexttext=contenttext, question=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")