'''agent that uses a combination of openai and duckduckgo to answer questions. '''
from langchain import LLMMathChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun


def ask_function_agent(query, temperature=0, model="gpt-3.5-turbo-0613"):
    '''ask the function agent a question.'''
    llm = ChatOpenAI(temperature=temperature, model=model, verbose=True)
    search = DuckDuckGoSearchRun()
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    tools = [
        Tool(
            name="Search",
            func=search.run(query),
            description='''
            useful for when you need to answer questions 
            about current events. You should ask targeted questions
            '''
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        )
    ]
    agent = initialize_agent(
        tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
    agent.run(query=query, verbose=True)
