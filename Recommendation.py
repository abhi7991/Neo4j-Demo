# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 20:24:44 2024

@author: abhis
"""

import matplotlib.pyplot as plt    
import pandas as pd
from neo4j import GraphDatabase
import os
import time   
import pandas as pd
from neo4j import GraphDatabase
import os
import time
from graphdatascience import GraphDataScience
wd = os.getcwd()

def read_params_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return [line.strip() for line in lines]   
#df = pd.read_csv(r"C:\Users\abhis\Desktop\NEU\DAMG7374 LLM w Knowledge Graph DB\Movie_Data\\"+"movies_metadata.csv")
uri, user, password = read_params_from_file(wd+"\\params.txt") 
driver = GraphDatabase.driver(uri, auth=('neo4j', '@Fd2556b9dd'), max_connection_lifetime=200)
gds = GraphDataScience(
    uri,
    auth = (user, password)
)       
#%%
request="""
MATCH (g:Genre)<-[:GENRE]-(m:Movie)
WITH g.name as genre,count(*) as total
RETURN  genre,total
ORDER BY total DESC
"""

result = {"label": [], "count": []}
for label in pd.DataFrame(gds.run_cypher("CALL db.labels()")).iloc[:,0]:
    query = f"MATCH (:`{label}`) RETURN count(*) as count"
    count = pd.DataFrame(gds.run_cypher(query)).iloc[0]['count']
    result["label"].append(label)
    result["count"].append(count)
nodes_df = pd.DataFrame(data=result)
nodes_df.sort_values("count")

nodes_df.plot(kind='bar', x='label', y='count', legend=None, title="Node Cardinalities")
plt.yscale("log")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#result = gds.run_cypher(request)#graph.run(request).to_data_frame()
#result.plot.barh(x='genre', y='total',figsize=(10,8))
#plt.show()

# Cypher query to count nodes

nodes_query = "MATCH (n) RETURN COUNT(n) AS total_nodes"
# Cypher query to count relationships
relationships_query = "MATCH ()-[r]->() RETURN COUNT(r) AS total_relationships"
# Plotting
total_nodes = gds.run_cypher(nodes_query).iloc[0,0]
total_relationships = gds.run_cypher(relationships_query).iloc[0,0]
labels = ['Nodes', 'Relationships']
counts = [total_nodes, total_relationships]


plt.bar(labels, counts, color=['blue', 'green'])
plt.xlabel('Element')
plt.ylabel('Count')
plt.title('Total Number of Nodes and Relationships')
plt.show()

nodes_query = """
MATCH (n)
RETURN labels(n) AS labels, COUNT(n) AS node_count
"""

# Cypher query to count relationships per type
relationships_query = """
MATCH ()-[r]->()
RETURN type(r) AS relationship_type, COUNT(r) AS relationship_count
"""

# Execute Cypher queries
node_results = gds.run_cypher(nodes_query)
relationship_results = gds.run_cypher(relationships_query)

# Plotting node counts per label
plt.figure(figsize=(10, 5))
plt.bar(node_results['labels'].apply(lambda x: x[0]), node_results['node_count'], color='blue')
plt.xlabel('Label')
plt.ylabel('Node Count')
plt.title('Number of Nodes per Label')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting relationship counts per type
plt.figure(figsize=(10, 5))
plt.bar(relationship_results['relationship_type'], relationship_results['relationship_count'], color='green')
plt.xlabel('Relationship Type')
plt.ylabel('Relationship Count')
plt.title('Number of Relationships per Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#%%
'''

Coreect one


1. We create a new GDS graph projection with whatever node properties and relationships we want

2. 


'''
del1 = "CALL gds.graph.drop('movies',false);"
del2 = "CALL gds.graph.drop('movies2',false);"

        
query3 = """CALL gds.graph.project('movies2', 
    {
        Movie: {properties: 'revenue'}
    },
    {
        GENRE: {orientation: 'UNDIRECTED'} 
    }
);

"""
query4 = """//Forming a KNN Graph
CALL gds.knn.write(
  'movies2',
  {
    nodeLabels:['Movie'] ,
    nodeProperties:'revenue',
    sampleRate: 0.5,
    deltaThreshold: 0.1,
    randomSeed: 42,
    concurrency: 1,
    writeProperty: 'score',
    writeRelationshipType: 'SIMILAR'
  }
)
YIELD similarityDistribution
RETURN similarityDistribution.mean AS meanSimilarity;
"""

for q in [del1,del2,query3,query4]:
    gds.run_cypher(q)
    
verify = 'match (n:Movie)-[k:SIMILAR]->(m:Movie) where n.name ="Star Wars" return n.name,m.name,k.score,n.id,m.id limit 100;'
result = gds.run_cypher(verify)    
#%%
del1 = "CALL gds.graph.drop('movies',false);"
del2 = "CALL gds.graph.drop('movies2',false);"
query1 = """CALL gds.graph.project('movies', 
              ['Movie','User','Genre'], 
              {RATING:{orientation: 'UNDIRECTED'},
               GENRE:{orientation: 'UNDIRECTED'}              
              }
            );"""

query2 = """CALL gds.fastRP.write(
          'movies',
          {
            nodeLabels:['Movie','User','Genre'], 
            relationshipTypes:['RATING','GENRE'],
            iterationWeights: [0.8,1.0],
            embeddingDimension: 10,
            writeProperty: 'embedding_fastrp',
            nodeSelfInfluence:10
          }
        )
        YIELD nodePropertiesWritten;"""
        
query3 = """CALL gds.graph.project('movies2', 
    {
        Movie: {properties: 'embedding_fastrp'},
        User: {properties: 'embedding_fastrp'}, 
        Genre: {properties: 'embedding_fastrp'} 
    },
    {
        RATING: {orientation: 'UNDIRECTED'},
        GENRE:{orientation: 'UNDIRECTED'}
    }
);

"""


query4 = """//Forming a KNN Graph
CALL gds.knn.write(
  'movies2',
  {
    nodeLabels:['Movie','User','Genre'] ,
    nodeProperties:'embedding_fastrp',
    sampleRate: 1.0,
    deltaThreshold: 0.0,
    randomSeed: 42,
    concurrency: 1,
    writeProperty: 'score',
    writeRelationshipType: 'SIMILAR'
  }
)
YIELD similarityDistribution
RETURN similarityDistribution.mean AS meanSimilarity;
"""


for q in [del1,del2,query3,query4]:
    gds.run_cypher(q)    
#%%
#result = gds.run_cypher(verify)#graph.run(request).to_data_frame()
verify = 'match (n:Movie)-[:SIMILAR]->(m:Movie) where n.name ="Toy Story" return n.name,m.name limit 100;'
#verify = """MATCH (n:Movie)<-[:Genre]-(g:Genre { name: "Fantasy" })-[:GENRE]-(similar:Movie)-[:SIMILAR]->(m:Movie)
#RETURN similar.name AS similarMovie, g.name AS genre
#LIMIT 100;
#"""
#verify = 'match (n:User)-[:SIMILAR]->(m:User) return n.userId,m.userId limit 100;'
#
#verify = '''MATCH (n:User)-[:SIMILAR]->(m:User)
#MATCH (n)-[:RATING]->(movie_n:Movie)
#MATCH (m)-[:RATING]->(movie_m:Movie)
#RETURN n.userId AS user_id,
#       m.userId AS similar_user_id,
#       COLLECT(DISTINCT movie_n.name) AS movies_rated_by_n,
#       COLLECT(DISTINCT movie_m.name) AS movies_rated_by_m
#LIMIT 100;
#
#'''

result = gds.run_cypher(verify)
#result['Common'] = result.apply(lambda x : list(set(x['movies_rated_by_n']).intersection(x['movies_rated_by_m'])),axis=1)
#%%

'''
Find Actors who an actor has not worked with
'''
q = """MATCH (tom:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(movie1:Movie)<-[:ACTED_IN]-(coActor:Person)-[:ACTED_IN]->(movie2:Movie)<-[:ACTED_IN]-(coCoActor:Person)
WHERE tom <> coCoActor
AND NOT (tom)-[:ACTED_IN]->(:Movie)<-[:ACTED_IN]-(coCoActor)
RETURN coCoActor.name limit 10"""
q = """MATCH (inputMovie:Movie {id: 862})<-[r:Genre]-(h:Genre)
WITH inputMovie, COLLECT (h) as inputGenres
MATCH (inputMovie)<-[r:RATING]-(User)-[o:RATING]->(movie)<-[:Genre]-(a:Genre) 
WITH  inputGenres,  r, o, movie, COLLECT(a) AS genres 
WHERE ALL(h in inputGenres where h in genres)
RETURN movie.title,movie.movieId, count(*) 
ORDER BY count(*) DESC"""

#q = """MATCH (m:Movie {name:'Toy Story'})<-[:RATING]-(u:User)-[:RATING]->(rec:Movie)
#RETURN distinct rec.name AS recommendation LIMIT 20"""

q = """//SIMILARITY 1
MATCH (s:SpokenLanguage{name:'English'})-[:LANGUAGE]->(inputMovie:Movie {id: 11})<-[:Genre]-(inputGenre:Genre)
WITH inputMovie, inputGenre.name AS inputGenres

MATCH (inputMovie)<-[r:RATING]-(user:User)-[o:RATING]->(movie:Movie)<-[:Genre]-(genre:Genre) 

WITH inputGenres, r, o, movie, genre.name AS movieGenres 
WHERE ALL(inputGenre IN inputGenres WHERE inputGenre IN movieGenres)

RETURN movieGenres,movie.name, AVG(r.rating) AS avgRating,count(r) as ratingCount
ORDER BY ratingCount DESC;
"""
q = """MATCH (inputMovie:Movie {id: 11})<-[:GENRE]-(inputGenre:Genre)
WITH inputMovie, inputGenre.name AS inputGenres

MATCH (inputMovie)<-[r:RATING]-(user:User)-[o:RATING]->(movie:Movie)<-[:GENRE]-(genre:Genre) 

WITH inputGenres, r, o, movie, COLLECT(genre.name) AS movieGenres 
WHERE ALL(inputGenre IN inputGenres WHERE inputGenre IN movieGenres)

RETURN movieGenres,movie.name, movie.id, COUNT(*) AS ratingCount
ORDER BY ratingCount DESC;
"""

q = """MATCH (s:SpokenLanguage{name:'English'})-[:LANGUAGE]->(inputMovie:Movie {id: 862})<-[:GENRE]-(inputGenre:Genre)
WITH inputMovie, inputGenre.name AS inputGenres

MATCH (inputMovie)<-[r:RATING]-(user:User)-[o:RATING]->(movie:Movie)<-[:GENRE]-(genre:Genre) 

WITH  inputGenres, r, o, movie, genre.name AS movieGenres,inputMovie.name as movieName 
WHERE ALL(inputGenre IN inputGenres WHERE inputGenre IN movieGenres)

RETURN movieGenres,movieName,movie.name, AVG(r.rating) AS avgRating,count(r) as ratingCount
ORDER BY ratingCount DESC;
"""
result = gds.run_cypher(q)
#%%