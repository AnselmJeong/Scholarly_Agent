from typing import Annotated, TypedDict
from IPython.display import Image, display, Markdown
from langchain_core.tools.base import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel

from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, add_messages
from langgraph.prebuilt import ToolNode, tools_condition


def _display_graph(graph):
    """Display the StateGraph using IPython's display function"""
    display(Image(graph.get_graph().draw_mermaid_png()))


CompiledStateGraph.draw = _display_graph


def _generate_chatbot_node(llm: BaseChatModel, prompt: str, tools: list[BaseTool] = None):
    """Generate a chatbot node that uses the provided LLM and tools"""
    if tools:
        llm_chain = prompt | llm.bind_tools(tools)
    else:
        llm_chain = prompt | llm

    def _node(state: dict):
        if "messages" not in state:
            raise ValueError("messages key is required in state")
        response = llm_chain.invoke(state["messages"])
        return {"messages": [response]}

    return _node


class State(TypedDict):
    messages: Annotated[list, add_messages]


def generate_tool_graph(llm: BaseChatModel, prompt: str, tools: list[BaseTool]):
    """Generate graph that uses the provided LLM and provided tools"""
    graph_builder = StateGraph(State)
    chatbot_node = _generate_chatbot_node(llm, prompt, tools)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_node("tools", ToolNode(tools))

    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    graph = graph_builder.compile()
    return graph
