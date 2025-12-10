from dotenv import load_dotenv
# from langchain.globals import set_verbose, set_debug
from langchain_groq.chat_models import ChatGroq
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

from agent.prompts import *
from agent.states import *
# from agent.tools import write_file, read_file, get_current_directory, list_files
load_dotenv()

llm = ChatGroq(model = "openai/gpt-oss-120b")



def planner_agent(state : dict) -> dict :
    user_prompt = state["user_prompt"]
    resp = llm.with_structured_output(Plan).invoke(
        planner_prompt(user_prompt))
    return {"plan": resp}





graph = StateGraph(dict)
graph.add_node("planner", planner_agent)
graph.set_entry_point("planner")


agent = graph.compile()


user_prompt = "create a simple calculator web application"


result = agent.invoke({"user_prompt": user_prompt})

print(result)