from typing import Optional, Type

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.retrievers.pubmed import PubMedRetriever
from langchain_core.tools import tool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from metapub.convert import pmid2doi

from dotenv import load_dotenv
import os

load_dotenv()


def _suggest_refined_query(user_input):
    """
    Generates a PubMed-optimized search query using LLM
    """
    # Initialize ChatOpenAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a biomedical research assistant expert in PubMed search strategies.
            Analyze the research question/topic given by the user.
            First, get the user's intent and then generate an appropriate search terms.
            Second, use proper PubMed syntax to convert the generated terms into a search query
            You may include: Boolean operators, MeSH terms, 
            [tiab] tags for title/abstract searches, and quotation marks for exact phrases. 
            Prioritize specificity and recall. Respond only with the search query itself.
         """,
        ),
        (
            "user",
            """Generate an effective PubMed search query for the topic : {input}""",
        ),
    ])

    try:
        # Create chain and invoke
        chain = prompt | llm
        response = chain.invoke({"input": user_input})
        return response.content.strip()

    except Exception as e:
        print(f"Error generating query: {e}")
        return None


class PubMedSearchInput(BaseModel):
    """Input schema for PubMed search"""

    query: str = Field(
        description="The search query for PubMed. Can be a research topic, question, or specific medical terms."
    )


class PubMedSearchTool(BaseTool):
    name: str = "pubmed_search"
    description: str = """Search PubMed for scientific papers on biomedical topics. 
    This tool can find recent research papers, clinical studies, and medical literature."""
    args_schema: Type[BaseModel] = PubMedSearchInput
    pm_retriever: PubMedRetriever = None
    refine_query: bool = False
    return_direct: bool = True

    def __init__(self, top_k_results: int = 10, refine_query: bool = False):
        super().__init__()
        if not os.getenv("NCBI_API_KEY"):
            raise ValueError("NCBI API key is required")
        self.pm_retriever = PubMedRetriever(top_k_results=top_k_results, api_key=os.getenv("NCBI_API_KEY"))
        self.refine_query = refine_query

    def _format_response(self, responses):
        papers = [
            {
                "pmid": paper.metadata["uid"],
                "doi": pmid2doi(paper.metadata["uid"]),
                "title": paper.metadata["Title"],
                "pub_date": paper.metadata["Published"],
                "abstract": paper.page_content,
            }
            for paper in responses
        ]

        markdown_text = "# PubMed Search Results\n\n"
        for paper in papers:
            markdown_text += f"## {paper['title']}\n"
            markdown_text += (
                f"**PMID**: {paper['pmid']} | **DOI**: {paper['doi']} | **Published**: {paper['pub_date']}\n\n"
            )
            markdown_text += f"**Abstract**:\n{paper['abstract']}\n\n"
            markdown_text += "---\n\n"  # Added separator between papers
        return markdown_text

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Execute the PubMed search"""
        if self.refine_query:
            query = _suggest_refined_query(query)
        responses = self.pm_retriever.invoke(query)
        return self._format_response(responses)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Asynchronously execute the PubMed search"""
        return self._run(query, run_manager)


if __name__ == "__main__":
    user_topic = "serious complications of 'Car-T cell' therapy for cancer"
    tool = PubMedSearchTool(api_key=os.getenv("NCBI_API_KEY"), refine_query=True)
    print(tool.invoke(user_topic))
