"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""

from typing import List, Optional
from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

# Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None  # lazy-init agent

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Classic retriever node"""
        docs = self.retriever.invoke(state["question"])
        return {
            "question": state["question"],
            "retrieve_docs": docs,
            "answer": state.get("answer", "")
        }

    def _build_tools(self):
        """Build retriever + wikipedia tools"""

        @tool
        def retriever(query: str) -> str:
            """Fetch passages from indexed corpus."""
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)

        @tool
        def wikipedia(query: str) -> str:
            """Search Wikipedia for general knowledge."""
            api_wrapper = WikipediaAPIWrapper()  # type: ignore
            wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
            return wiki.run(query)

        return [retriever, wikipedia]

    def _build_agent(self):
        """ReAct agent with tools"""
        tools = self._build_tools()
        self._agent = create_agent(self.llm, tools=tools)

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer using ReAct agent with retriever + wikipedia.
        """
        if self._agent is None:
            self._build_agent()

        assert self._agent is not None, "Agent must be built before use"
        result = self._agent.invoke({"messages": [HumanMessage(content=state["question"])]})

        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        return {
            "question": state["question"],
            "retrieve_docs": state["retrieve_docs"],
            "answer": answer or "Could not generate answer."
        }
