import datetime
import functools
import inspect
import logging
from typing import TypedDict

from langchain_core.runnables.base import RunnableLike
from langgraph.graph import StateGraph

from myscalekb_agent.schemas.agent_metadata import AgentMetadata

logger = logging.getLogger(__name__)


def node(func):
    """Adds a new node to the graph.
    . code-block:: python

        @node
        async def run_query(self, data):
            # code logic

    Adds the following cross-cutting concerns:
    - Logging start and end of node execution
    - Performance timing
    - Agent metadata creation
    - Consistent configuration handling
    """

    @functools.wraps(func)
    async def wrapper(self, data):
        trace_id = data.get("trace_id", "unknown")
        agent_metadata = AgentMetadata(name=self.name(), step=func.__name__)

        logger.info("QueryTrace[%s] Beginning to run %s step.", trace_id, func.__name__)
        start_time = datetime.datetime.now()

        try:
            # Execute the original function
            data["agent_metadata"] = agent_metadata
            result = await func(self, data)

            duration = (datetime.datetime.now() - start_time).total_seconds()

            logger.info(
                "QueryTrace[%s] Completed running %s step, duration: %.3f seconds",
                trace_id,
                func.__name__,
                duration,
            )

            if not result:
                return {"agent_metadata": agent_metadata}
            elif isinstance(result, dict):
                result["agent_metadata"] = agent_metadata

            return result

        except Exception as e:
            logger.error("QueryTrace[%s] Error in %s step: %s", trace_id, func.__name__, str(e))
            raise

    wrapper._is_node = True
    return wrapper


def edge(target_node: str):
    """Annotate on the source node to establish a unidirectional edge to the target node.
    . code-block:: python

        @node
        @edge(target_node="summary_contexts")
        async def retrieve_contexts(self, data):
            # code logic

        @node
        @edge(target_node=GraphBuilder.END)
        async def summary_contexts(self, data):
            # code logic

    """

    def decorator(func):
        func._is_edge = True
        func._target = target_node
        return func

    return decorator


def conditional_edge(path: RunnableLike, path_map: dict):
    """Annotate on the source node and define different conditions to reach different nodes.
    . code-block:: python

        @conditional_edge(source_node="run_query", path_map={"continue": "retrieve_contexts", "end": GraphBuilder.END})
        def should_continue(self, data):
            if isinstance(data["outcome"], AgentFinish):
                return  "continue"
            return "continue"

    """

    def decorator(func):
        func._is_conditional_edge = True
        func._path = path
        func._path_map = path_map
        return func

    return decorator


def entry(func):
    """Mark a node as entry point in Graph.
    A Graph can only have one entry. This annotation cannot be used with @conditional_entry
    . code-block:: python

        @node
        @entry
        async def run_query(self, data):
            # code logic

    """
    func._is_entry = True
    return func


def conditional_entry(path_map: dict):
    """Mark an edge as entry point in Graph.
    Some special graphs need to determine which node to branch to at the entrance.
    . code-block:: python

        @conditional_entry(path_map={"form": "init_form", "query": "run_query"})
        async def entry(self, data):
            # code logic

    """

    def decorator(func):
        func._is_conditional_entry = True
        func._path_map = path_map
        return func

    return decorator


class GraphBuilder:

    END = "__end__"

    def _build_graph(self, state_definition: TypedDict):
        workflow = StateGraph(state_definition)

        for name, member in inspect.getmembers(self.__class__, predicate=inspect.isfunction):
            method = getattr(self.__class__, name)
            print(f"Checking method {name}: {method.__dict__}")

            if hasattr(method, "_is_node"):
                bound_method = method.__get__(self, self.__class__)
                workflow.add_node(bound_method)
                print(f"Added node: {name}")

            if hasattr(method, "_is_edge"):
                target = method._target
                workflow.add_edge(name, target)
                print(f"Added edge from {name} to {target}")

            if hasattr(method, "_is_conditional_edge"):
                workflow.add_conditional_edges(source=name, path=method._path, path_map=method._path_map)
                print(f"Added conditional edge from {name}")

            if hasattr(method, "_is_entry"):
                workflow.set_entry_point(name)
                print(f"Set entry point: {name}")

            if hasattr(method, "_is_conditional_entry"):
                bound_method = method.__get__(self, self.__class__)
                path_map = method._path_map
                workflow.set_conditional_entry_point(path=bound_method, path_map=path_map)
                print(f"Added conditional entry {path_map}")

        graph = workflow.compile()
        print("Build graph successfully", graph)
        return graph
