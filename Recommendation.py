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
driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=200)
gds = GraphDataScience(
    uri,
    auth = (user, password)
)       
request="""
MATCH (g:Genre)<-[:HAS_GENRE]-(m:Movie)
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

result = gds.run_cypher(request)#graph.run(request).to_data_frame()
result.plot.barh(x='genre', y='total',figsize=(10,8))
plt.show()

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

del1 = "CALL gds.graph.drop('movies',false);"
del2 = "CALL gds.graph.drop('movies2',false);"
query1 = """CALL gds.graph.project('movies', 
              ['Movie','User','Genre'], 
              {RATING:{orientation: 'REVERSE',properties:'rating'},HAS_GENRE:{orientation: 'REVERSE'}}
            );"""

query2 = """CALL gds.fastRP.write(
          'movies',
          {
            nodeLabels:['Movie','Genre','User'],
            relationshipTypes:['RATING'],
            iterationWeights: [0.8,1.0],
            relationshipWeightProperty:'rating',
            embeddingDimension: 200,
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
        RATING: {orientation: 'REVERSE', properties: 'rating'},
        HAS_GENRE: {orientation: 'REVERSE'}
    }
);

"""
query4 = """//Forming a KNN Graph
CALL gds.knn.write(
  'movies2',
  {
    nodeLabels:['Movie','User'] ,
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

verify = 'match (n:User)-[:SIMILAR]->(m:User) return n.id,m.id limit 100;'

verify = '''MATCH (n:User)-[:SIMILAR]->(m:User)
MATCH (n)-[:RATING]->(movie_n:Movie)
MATCH (m)-[:RATING]->(movie_m:Movie)
RETURN n.id AS user_id,
       m.id AS similar_user_id,
       COLLECT(DISTINCT movie_n.id) AS movies_rated_by_n,
       COLLECT(DISTINCT movie_m.id) AS movies_rated_by_m
LIMIT 100;

'''
for q in [del1,del2,query1,query2,query3,query4]:
    gds.run_cypher(q)
result = gds.run_cypher(verify)#graph.run(request).to_data_frame()


result['Common'] = result.apply(lambda x : list(set(x['movies_rated_by_n']).intersection(x['movies_rated_by_m'])),axis=1)
