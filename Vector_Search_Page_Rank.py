# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:08:07 2024

@author: abhis
"""

import pandas as pd
from neo4j import GraphDatabase
import os
import time
wd = os.getcwd()

'''

User Ratings 

'''


df = pd.read_csv(r"C:\Users\abhis\Desktop\NEU\DAMG7374 LLM w Knowledge Graph DB\Movie_Data\\"+"ratings.csv")
len(df.iloc[:,0].unique())
a =  df.groupby("userId")['movieId'].count().reset_index()

print(a)
del df

df = pd.read_csv(r"C:\Users\abhis\Desktop\NEU\DAMG7374 LLM w Knowledge Graph DB\Movie_Data\\"+"ratings_small.csv")
len(df.iloc[:,0].unique())
a =  df.groupby("userId")['movieId'].count().reset_index()
print(a)



'''

Performing Vector Search

'''
def read_params_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return [line.strip() for line in lines]   
df = pd.read_csv(r"C:\Users\abhis\Desktop\NEU\DAMG7374 LLM w Knowledge Graph DB\Movie_Data\\"+"movies_metadata.csv")
uri, user, password = read_params_from_file(wd+"\\params.txt") 
driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=200)
#driver.set_database('Csv2Graph')

ids = df['id'].to_list()[:200]
for id1 in ids:
    time.sleep(1)
    query = "MATCH (m:Movie) WHERE m.id = "+str(id1)+" CALL db.index.vector.queryNodes('overview_embeddings', 10, m.embedding) YIELD node AS similarMovie, score RETURN m.name AS movieName, similarMovie.name AS similarMovieName, score;"
#    query = "MATCH (m:Movie) CALL db.index.vector.queryNodes('overview_embeddings', 10, m.embedding) YIELD node AS similarMovie, score RETURN m.name AS movieName, similarMovie.name AS similarMovieName, score;"
    
    with driver.session() as session:
        vals = session.run(query)
        results_list = [dict(record) for record in vals]
        print(results_list)
driver.close()

'''

Peforming Page Rank

'''

drop = "CALL gds.graph.drop('myGraph',false);"   
create = """CALL gds.graph.project(
              'myGraph',
              ['User', 'Movie'],
              {
                RATING: {
                  type: 'RATING',
                  orientation: 'REVERSE' , type:'*'                 
                }
              }
            );"""    
#create = """
#    CALL gds.graph.project('
#      'myGraph',
#      {
#        Movie: {properties: 'name'},
#       }
#      },
#      ['RATED']
#)"""    
pageRank = """MATCH (user:User {id: 15})
            CALL gds.pageRank.stream('myGraph', {
              maxIterations: 20,
              dampingFactor: 0.85,
              sourceNodes: [user]
            })
            YIELD nodeId, score
            
            MATCH (movie:Movie)
            WHERE id(movie) = nodeId
            
            RETURN movie.id AS movieId, movie.name AS movieName, score
            ORDER BY score DESC;"""   


pageRank = """CALL gds.pageRank.stream('myGraph', {
              maxIterations: 20,
              dampingFactor: 0.85
            })
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).id AS movieId, gds.util.asNode(nodeId).name AS movieName, score
            ORDER BY score DESC;
        """            
with driver.session() as session:

    try:
        session.run(drop)
#        session.run("CALL gds.graph.drop('myGraph')") 
    except:
#        session.run("DROP GRAPH myGraph")
        print("CANT")
        pass
    session.run(create)

    vals = session.run(pageRank)
    results_list = [dict(record) for record in vals]
    df = pd.DataFrame(results_list)
    movie_ids_list = df['movieId'].tolist()
    movie_names_query = """
    MATCH (movie:Movie)
    WHERE movie.id IN $movieIds 
    RETURN movie.id AS movieId, movie.name AS movieName
    """    
    result = session.run(movie_names_query, movieIds=movie_ids_list)
    movie_names_list = [dict(record) for record in result]
    movie_names_df = pd.DataFrame(movie_names_list)    
#%%
'''

Querying Database with custom queries 

'''    
    
    
with driver.session() as session:
 
    vals = session.run("MATCH (u:User {id: 60})-[:RATING]->(m:Movie) RETURN m.id AS movieId, m.name AS movieName, u.rating AS userRating;")

    results_list = [dict(record) for record in vals]
                   
    umovie = pd.DataFrame(vals)
    print(umovie)



#%%
    
'''

Loading Json Column into Neo4j


'''    
wd = r"C:\Users\abhis\.Neo4jDesktop\relate-data\dbmss\dbms-270e66cc-c52d-46b5-9148-c8e81def25cf\import"    
    

df = pd.read_csv(wd+"\movies_metadata.csv")
df['production_companies'].apply(lambda x : (len(str(x)) < 6))

df = df[df['production_companies'].apply(lambda x:((x!='[]') & (x!='False') & (x!='True')))]
df.to_csv(wd+"\clean_movies_metadata.csv",index=False)

query = """
LOAD CSV WITH HEADERS FROM 'file:///clean_movies_metadata.csv' AS row

// Handle errors during JSON parsing
WITH row, apoc.convert.fromJsonList(row['production_companies']) AS productionCompaniesList

// Check if the parsed JSON is valid
WITH row, productionCompaniesList,
     CASE
       WHEN productionCompaniesList IS NOT NULL AND size(productionCompaniesList) > 0
       THEN productionCompaniesList
       ELSE null
     END AS validProductionCompaniesList

// Create nodes and relationships if the JSON is valid
FOREACH (company IN validProductionCompaniesList |
  MERGE (m:Movie {id: toInteger(row['id'])})
  MERGE (p:ProductionCompany {name: company.name}) ON CREATE SET p.id = toInteger(company.id)
  MERGE (m)-[:PRODUCED_BY]->(p)
);


"""
import pandas as pd
from neo4j import GraphDatabase
import os
import time
wd = os.getcwd()
def read_params_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return [line.strip() for line in lines]   
    
uri, user, password = read_params_from_file(wd+"\\params.txt")     
driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=200)

with driver.session() as session:
    vals = session.run(query)
  
#%%

    