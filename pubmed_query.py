from langchain_openai import ChatOpenAI

from langchain.prompts import ChatPromptTemplate


def suggest_pubmed_query(user_input):
    """
    Generates a PubMed-optimized search query using LLM
    """
    # Initialize ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a biomedical research assistant expert in PubMed search strategies."),
        (
            "user",
            """Generate an effective PubMed search query for the following research question/topic. 
Use proper PubMed syntax including: Boolean operators, MeSH terms when appropriate, 
[tiab] tags for title/abstract searches, and quotation marks for exact phrases. 
Prioritize specificity and recall. Respond only with the search query itself.

Topic: {input}""",
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


if __name__ == "__main__":
    user_topic = "serious complications of 'Car-T cell' therapy for cancer"
    search_query = suggest_pubmed_query(user_topic)
    print(search_query)
