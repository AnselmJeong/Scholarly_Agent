from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.pubmed_tool import PubMedSearchTool
from utils.langgraph_utils import generate_tool_graph
from config import CONFIG

GENAI_MODEL = CONFIG["llm"]["gemini_model"]
PROMPT = CONFIG["prompts"]["pubmed_prompt"]

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        PROMPT,
    ),
    (
        "user",
        "{input}",
    ),
])

llm = ChatGoogleGenerativeAI(model=GENAI_MODEL)

tools = [PubMedSearchTool(top_k_results=20)]
graph = generate_tool_graph(llm, prompt, tools)
