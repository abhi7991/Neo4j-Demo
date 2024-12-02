import os
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.chains import LLMChain
import json
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field


load_dotenv(override=True)


NEO4J_USERNAME = os.getenv('NEO4J_USER')
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
DATABASE = os.getenv('NEO4J_DATABASE')
CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j Cypher translator who understands the question in english and convert to Cypher strictly based on the Neo4j Schema provided and following the instructions below:
<instructions>
* Use aliases to refer the node or relationship in the generated Cypher query
* Generate Cypher query compatible ONLY for Neo4j Version 5
* Do not use EXISTS, SIZE keywords in the cypher. Use alias when using the WITH keyword
* Use only Nodes and relationships mentioned in the schema
* Always enclose the Cypher output inside 3 backticks (```)
* Always do a case-insensitive and fuzzy search for any properties related search. Eg: to search for a Company name use `toLower(c.name) contains 'neo4j'`
* Cypher is NOT SQL. So, do not mix and match the syntaxes
</instructions>

Strictly use this Schema for Cypher generation:
<schema>
{schema}
</schema>

The samples below follow the instructions and the schema mentioned above. So, please follow the same when you generate the cypher:
<samples>
Human: Which actors have most movies? What is the total number of movies they acted in ?
Assistant: ```MATCH (p:Person) -[a:ACTED_IN]->(m:Movie) RETURN p.name as Actor, count(distinct m) as totalmovies ORDER BY totalmovies DESC LIMIT 10```

Human: Which 5 production houses produced most movies? How many movies?
Assistant: ```MATCH (c:ProductionCompany)-[:PRODUCED_BY]->(m:Movie) WITH c.name as ProductionHouse, count(m) AS numMovies RETURN ProductionHouse, numMovies ORDER BY numMovies DESC LIMIT 5;```

Human: What are the top 10 grossing movies for Warner Bros.?
Assistant: ```MATCH (c:ProductionCompany)-[:PRODUCED_BY]->(m:Movie) WHERE toLower(c.name) CONTAINS "warner bros." RETURN DISTINCT c.name AS ProductionHouse, m.name AS Movies, m.revenue AS Revenue ORDER BY Revenue DESC LIMIT 10;```

Human: Which 5 actor pairs did most movies by Warner Bros together?
Assistant: ```MATCH (:ProductionCompany {{name: 'Warner Bros.'}})<-[:PRODUCED_BY]-(m:Movie)<-[:ACTED_IN]-(a1:Person) WITH m, a1 MATCH (m)<-[:ACTED_IN]-(a2:Person) WHERE id(a1) < id(a2) RETURN a1.name AS Actor1, a2.name AS Actor2, count(DISTINCT m) AS SharedMovies ORDER BY SharedMovies DESC LIMIT 5```

Human: What are the top 5 genres with most movies?
Assistant: ```MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre) RETURN g.name as Genre, count(Distinct m) as numMovies ORDER BY numMovies DESC LIMIT 5```

Human: Who directed Inception?
Assistant: ```MATCH (p:Person)-[:CREWED_IN {{character: 'Directing'}}]->(m:Movie {{name: 'Inception'}}) RETURN p.name as Director```

Human: Give me the plot or story of the movie Forrest Gump?
Assistant: ```MATCH (m:Movie {{name: 'Forrest Gump'}}) RETURN m.overview as Plot```

</samples>

Human: {question}
Assistant: 
"""


CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=['schema','question'], validate_template=True, template=CYPHER_GENERATION_TEMPLATE
)



graph = Neo4jGraph(
    url=NEO4J_URI, 
    username=NEO4J_USERNAME, 
    password=NEO4J_PASSWORD,
    database = DATABASE
)

llm = ChatOpenAI(temperature=0,openai_api_key=os.getenv('OPENAI_API_KEY'))

chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    verbose=False,
    return_direct=True
)

class SearchInput(BaseModel):
    que: str = Field(...,description="this would be question by the user about what the movies are, who are the actors acting ,what the plot of a movie is ,and other attributres related to movies it will be used to query but not for recommendation")
    
    
    
@tool(args_schema=SearchInput)
def chat(que:str)->str:
    
    """Conversing with the Knowledge graph of movies, and will generate cypher queries to get answers"""
    
    r = chain.invoke(que)
    # print(r)
    summary_prompt = """Human: 
    Fact: {result}

    * Summarise the above fact as if you are answering this question "{query}"
    * When the fact is not empty, assume the question is valid and the answer is true
    * Do not return helpful or extra text or apologies
    * Just return summary to the user. DO NOT start with Here is a summary
    * List the results in rich text format if there are more than one results
    Assistant:
    """
    summary_prompt_template = PromptTemplate(input_variables=['query','result'], validate_template=True, template=summary_prompt)
    llmchain = LLMChain(llm=llm, prompt=summary_prompt_template)
    summary = llmchain.invoke({'query':r['query'],'result':json.dumps(r['result'])})
    
    return summary['text']

## Need to add output parsers