import matplotlib.pyplot as plt    
import pandas as pd
from neo4j import GraphDatabase
import sys

import os
import requests
import json
import requests
import numpy as np
from graphdatascience import GraphDataScience
wd = os.getcwd()
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
from . import graph_build,create_plot_embeddings
#Uncomment
from modules import node_similarity,qa_bot,plot_vector_search 
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.agent import AgentFinish
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents import AgentExecutor


load_dotenv(override=True)

openai.api_key = os.environ['OPENAI_API_KEY']
database = 'neo4j' 
uri, user, password = os.getenv('NEO4J_URI'), os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD')

gds = GraphDataScience(
    uri,
    auth = (user, password), database=database
)       

def graph_intit():
    '''
    One time process - Incase starting from scratch use this to create graph, and embeddings for plot
    # '''
    graph_build.create_movie_graph()
    create_plot_embeddings.create_plot_embeddings()
    
   
def chat_bot(query):
    '''
    Chatbot with functions to answer questions related to the movies
    '''
    tools=[qa_bot.chat,node_similarity.getSimilar,plot_vector_search.vectorSearch]
    functions = [format_tool_to_openai_function(f) for f in tools]
    model = ChatOpenAI(temperature=0).bind(functions=functions)
    memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    chain = RunnablePassthrough.assign(
        agent_scratchpad = lambda x: format_to_openai_functions(x["intermediate_steps"])
    ) | prompt | model | OpenAIFunctionsAgentOutputParser()
    qa = AgentExecutor(agent=chain, tools=tools, verbose=True, memory=memory)
    result = qa.invoke({"input": query})
    answer = result['output'] 
    return answer


def create_user(email,password):
    '''
    New User Creation
    '''

    query = """MATCH (u:User)
            WITH MAX(u.userId) AS maxUserId
            CREATE (newUser:User {
                userId: maxUserId + 1,
                email: $email,
                password: $password
            })
            RETURN newUser.userId"""
            
    params={
        "email": email,
        "password": password
        }
                    
    gds.run_cypher(query,params = params).iloc[0,0]
    
def get_user(email,password):
    '''
    Get User Details
    '''
    query = """MATCH (u:User {email: $email, password: $password})
            RETURN u.userId AS userId"""
            
    params={
        "email": email,
        "password": password
        }
                    
    user_id = gds.run_cypher(query,params = params)
    # print(user_id.userId.iloc[0])
    if not user_id.empty:
        return True
    else:
        return False
    
def check_user(email):
    '''
    Check if user already exists
    '''
    query = """MATCH (u:User {email: $email})
            RETURN u.userId AS userId"""
            
    params={
        "email": email  
        }
    
    user_id = gds.run_cypher(query,params = params)
    
    if not user_id.empty:
        return True
    else:
        return False
    
    
def get_sample_movies(email):
    '''
    Top Movies based on the users who have interacted with the genres and popularity of the
    movies inside the Genres
    '''

    query = """MATCH (g:Genre)-[:GENRE]->(m:Movie)<-[:RATING]-(u:User)
            WITH g, COUNT(DISTINCT u) AS user_count
            ORDER BY user_count DESC
            LIMIT 10
            MATCH (genre:Genre {name: g.name})-[:GENRE]->(movie:Movie)
            WHERE NOT EXISTS {
                MATCH (movie)<-[:RATING]-(user:User {email: $email})
            }
            WITH genre.name AS Genre, movie.name AS TopMovies, movie.popularity AS Popularity, movie.id AS movieId
            ORDER BY genre.name, Popularity DESC
            WITH Genre, COLLECT({movieName: TopMovies, popularity: Popularity, movieId: movieId})[0..10] AS genreMovies
            RETURN Genre, genreMovies
            LIMIT 10;
            """
    params = {"email": email}
    df = pd.concat([pd.DataFrame(x) for x in gds.run_cypher(query,params).iloc[:,1]])
    df.drop_duplicates('movieId',inplace=True)
    df = df.sample(n=50, random_state=42)  # Adjust `n` as needed, and `random_state` for reproducibility
    return df


def resetRecommendation(email,password):
    
    query = "MATCH (u:User {email: $email, password:$password})-[r]-() DELETE r"
    params = {"email":email, "password":password}
    
    gds.run_cypher(query, params = params)    


def getImage(movieId):
    
    '''
    Get Images using the TMDB API
    '''
    #url = f"https://api.themoviedb.org/3/movie/{movieId}/images"+"?api_key=1a23dabc3e8f21345d6e1efb8b47db48"    
    url = f"https://api.themoviedb.org/3/movie/{movieId}/images"+"?api_key="+os.getenv('MOVIE_API_KEY')    
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer "+os.getenv('MOVIE_API_TOKEN')
    }    
    response = requests.get(url) 
    vals = json.loads(response.text)['posters']    
    vals = ['http://image.tmdb.org/t/p/w185/' + x['file_path'] if x['iso_639_1']=='en' else 'http://image.tmdb.org/t/p/w185/' + x['file_path'] for x in vals]
    return vals[0]

def save_preferences(preference_list, email,password):
    '''
    Output of this is a Genre -> list of dicts with each dict containing MovieId, 
    Name and popularity Now when he selects the movie the backend will have the 
    id of the movie and the user associcated with it loop over the movies he 
    selected and tag it to the user
    '''
    
    for movieId in preference_list:
        query = """MATCH (m:Movie {id: TOINTEGER($movieId)})
        MATCH (pc:User {email: $email, password: $password})
        MERGE (pc)-[r:RATING { rating: 0 }]->(m)"""        
        
        params = {"movieId":movieId, "email":email, "password":password}
        
        gds.run_cypher(query, params = params)
        
def generate_recommendations(email,password):
    '''
    Similarity of this user
    '''    
    
    del1 = "CALL gds.graph.drop('users',false);"
    gds.run_cypher(del1)
    query1 = """CALL gds.graph.project('users', 
                ['Movie','User','Genre','Person'], 
                {RATING:{properties:'rating',orientation:'reverse'},GENRE:{orientation:'reverse'},ACTED_IN:{orientation:'reverse'}, CREWED_IN:{orientation:'reverse'}}
                );"""

    gds.run_cypher(query1)    
    
    query = """MATCH (u:User {email: $email, password: $password})-[:RATING]->(m:Movie)
            WITH collect(id(m)) AS sourceNodeId
            CALL gds.nodeSimilarity.filtered.stream('users',{
                nodeLabels:['Movie','User','Genre','Person'],
                relationshipTypes:['RATING','GENRE','ACTED_IN','CREWED_IN'],
                sourceNodeFilter: sourceNodeId,
                targetNodeFilter:'Movie'
            })
            YIELD node1, node2, similarity
            RETURN gds.util.asNode(node1).name AS Movie1, gds.util.asNode(node2).name AS Movie2,gds.util.asNode(node2).id as movieId , similarity
            ORDER BY similarity DESCENDING, Movie1, Movie2"""

    similarity = gds.run_cypher(query, params = {"email":email, "password":password})
    similarity.drop_duplicates("movieId",inplace=True)        

    similarity['posterLink'] = similarity['movieId'].apply(lambda x : getImage(x))  
    
    return similarity

