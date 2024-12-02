from modules.relationship import getRelationship
import os
from graphdatascience import GraphDataScience
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field
from dotenv import load_dotenv
load_dotenv(override=True)


        
# uri, user, password = read_params_from_file(wd+"\\params.txt") 
uri, user, password,database = os.getenv('NEO4J_URI'), os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'), os.getenv('NEO4J_DATABASE')
gds = GraphDataScience(
    uri,
    auth = (user, password), database=database
)       
class SearchInput(BaseModel):
    question: str = Field(...,description="The question is based on a user's sentiment of what he wants to watch. The user will not ask about the plot or the story of a film. It essentially would not have enties or names, but just a general statement or question surrounding the movies description")
        
@tool(args_schema=SearchInput)
def vectorSearch(question: str) -> list:
    """
    Find movies based on the plot or genre of the movie using vector search on the question asked by the user on the plot or genre of the movie.
    """
    
    relationship = getRelationship(question)
    params={"openAiApiKey":os.getenv('OPENAI_API_KEY'),
            "openAiEndpoint": 'https://api.openai.com/v1/embeddings',
            "question": question,
            "top_k": 10}
    query = """
        WITH genai.vector.encode(
            $question, 
            "OpenAI", 
            {
              token: $openAiApiKey,
              endpoint: $openAiEndpoint
            }) AS question_embedding
        CALL db.index.vector.queryNodes(
            'overview_embeddings2', 
            $top_k, 
            question_embedding
        ) YIELD node AS movie, score
        WHERE movie.name IS NOT NULL """ + relationship + """
        RETURN movie.name, movie.overview, score
        """
    
    # print(query)
    df = gds.run_cypher(query, 
        params = params)
    
    x_list = df['movie.name'].head().tolist()
    
    return x_list
    