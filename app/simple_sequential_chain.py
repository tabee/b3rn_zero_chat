""" This is an a sequential chain. It runs two chains in sequence. """
import os
from dotenv import load_dotenv
import langchain
from langchain.llms import OpenAI
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.cache import SQLiteCache
from tools import vectorstore
from ai import chainbroker
os.environ["OPENAI_API_KEY"] = os.getenv(load_dotenv() and "OPEN_API_KEY")
langchain.llm_cache = SQLiteCache(
    database_path="/workspaces/b3rn_zero_chat/data/cache/langchain.db"
    )

def _get_promptoptimizer_chain():
    """ This is a chain that optimizes a prompt."""
    llm = OpenAI(temperature=.7)
    template = """
    Als Experte für Prompt Engineering ist es deine Aufgabe, 
    den folgenden Prompt zu optimieren. Ziel ist es, 
    die Klarheit und Präzision des Prompts zu verbessern, um 
    eine qualitativ hochwertigere Antwort zu generieren, als 
    sie mit dem ursprünglichen Prompt möglich wäre. Bitte beachte, 
    dass deine Optimierung speziell auf ein schweizerisches Publikum (Privatpersonen)
    abzielen soll.

    Prompt: {base_prompt}

    Optimierter Prompt:"""
    prompt_template = PromptTemplate(
        input_variables=["base_prompt"], template=template)
    return LLMChain(llm=llm, prompt=prompt_template)


def _get_promptpopulate_chain():
    """ This is a chain populate one question into 2-5 similar questions."""
    llm = OpenAI(temperature=.7)
    template = """
    Erstelle als Prompt Engineering Experte drei thematisch äquivalente Prompts, 
    die, wenn sie gestellt werden, zur gleichen Antwort führen wie die ursprüngliche Frage.
    Ziel ist es, die Klarheit und Präzision des Prompts zu verbessern, um 
    eine qualitativ hochwertigere Antwort zu generieren, als 
    sie mit dem ursprünglichen Prompt möglich wäre. Bitte beachte, 
    dass deine Optimierung speziell auf ein schweizerisches Publikum (Privatpersonen)
    abzielen soll.

    Prompt:
    {prompt}
    """
    prompt_template = PromptTemplate(
        input_variables=["prompt"], template=template)
    return LLMChain(llm=llm, prompt=prompt_template)


def _get_promptselector_chain():
    llm = OpenAI(temperature=.2)
    template = """
    Wähle als Prompt Engineering Experte aus folgenden Prompts, 
    das, welches auf welche die beste Antwort führt. Wenn es noch nicht der Fall ist,
    passe das Prompt so an, dass von der schweizerischen Bundesverwaltung beantwortet werden kann.

    3 Prompts:
    {prompt}

    Beste Prompt:
    """
    prompt_template = PromptTemplate(
        input_variables=["prompt"], template=template)
    return LLMChain(llm=llm, prompt=prompt_template)


def query_optimizer(query):
    """ This is the main function that runs the subsequent chains."""
    promptoptimizer_chain = _get_promptoptimizer_chain()
    review_chain = _get_promptpopulate_chain()
    selector_chain = _get_promptselector_chain()

    # This is the overall chain where we run these two chains in sequence.
    overall_chain = SimpleSequentialChain(
        chains=[promptoptimizer_chain, review_chain, selector_chain],
        verbose=True
    )
    return overall_chain.run(query)


if __name__ == "__main__":
    store = vectorstore.load_vectorstore()
    base_query = "Wie viel Familienzulage erhalte ich?"
    query = query_optimizer(base_query)

    doc_query = query + " und " + base_query
    result = chainbroker.conversational_retrieval_chain(
        query=query,
        vectorstore=store
    )
    print(result)
