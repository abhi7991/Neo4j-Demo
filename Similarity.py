# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 18:06:21 2024

@author: abhis
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
#df = pd.read_csv(r"C:\Users\abhis\Desktop\NEU\DAMG7374 LLM w Knowledge Graph DB\Movie_Data\\"+"movies_metadata.csv")
uri, user, password = read_params_from_file(wd+"\\params.txt") 
driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=200)

query0 = "CALL gds.graph.drop('movies',false);"

query1 = """CALL gds.graph.project('movies', 
              ['Movie','User','Genre','SpokenLanguage'], 
              {
                RATING:{orientation: 'UNDIRECTED',properties: 'rating'},
                HAS_GENRE:{orientation: 'UNDIRECTED'},
                HAS_SPOKEN_LANGUAGE:{orientation:'UNDIRECTED'}
              }
            );"""

query1 = """CALL gds.graph.project('movies', 
              ['Movie','Person'], 
              {
                ACTED_IN:{orientation: 'UNDIRECTED'}
              }
            );"""


query2 = """CALL gds.fastRP.mutate(
          'movies',
          {
            embeddingDimension: 100,
            randomSeed: 42,
            mutateProperty: 'similarities',
            embeddingDimension: 4,
            iterationWeights: [1, 1, 1, 1]
          }
        )
        YIELD nodePropertiesWritten;"""
        
query = """MATCH (m1:Movie {name: "Space Jam"})-[r:SIMILAR]-(m2:Movie)-[:HAS_SPOKEN_LANGUAGE]->(language:SpokenLanguage {name: 'English'})
RETURN m1.name AS movie1, m2.name AS similar_movie, r.score AS similarity
ORDER BY similarity DESCENDING, movie1, similar_movie;
"""
query = """MATCH (m1:Movie)-[:HAS_GENRE]->(:Genre {name: 'Fantasy'})<-[:HAS_GENRE]-(m2:Movie)-[r:SIMILAR]-(m3:Movie)-[:HAS_SPOKEN_LANGUAGE]->(language:SpokenLanguage {name: 'English'})
RETURN m1.name AS movie1, m3.name AS similar_movie, r.score AS similarity
ORDER BY similarity DESCENDING, movie1, similar_movie LIMIT 10;
"""
similar_movies = """CALL gds.nodeSimilarity.filtered.stream('movies',{nodeLabels:['Movie','Genre'],relationshipTypes:['HAS_GENRE'],sourceNodeFilter:'Movie' , targetNodeFilter:'Movie'})
YIELD node1, node2, similarity
WHERE gds.util.asNode(node1).name = 'Space Jam'
RETURN gds.util.asNode(node1).name AS node1Name, gds.util.asNode(node2).name AS node2Name, similarity;
"""

actors = """//Actors who have not acted with eachother
CALL gds.nodeSimilarity.stream('movies',{nodeLabels:['Movie','Person'],relationshipTypes:['ACTED_IN'],bottomK
:1})
YIELD node1, node2, similarity
RETURN gds.util.asNode(node1).name AS node1Name, gds.util.asNode(node2).name AS node2Name, similarity;"""
#
#""" 

with driver.session() as session:
    session.run(query0)
    session.run(query1)
    session.run(query2)

    vals = session.run(query)
    results_list = [dict(record) for record in vals]
#    print(results_list)
    df = pd.DataFrame(results_list).drop_duplicates()        