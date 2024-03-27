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
              ['Movie','User'], 
              {RATING:{orientation: 'UNDIRECTED',properties:'rating'}}
            );"""

query2 = """CALL gds.fastRP.write(
          'movies',
          {
            embeddingDimension: 5,
            writeProperty: 'embedding_fastrp'
          }
        )
        YIELD nodePropertiesWritten;"""
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
#query = """MATCH (m1:Movie)-[:HAS_GENRE]->(:Genre {name: 'Fantasy'})<-[:HAS_GENRE]-(m2:Movie)-[r:SIMILAR]-(m3:Movie)-[:HAS_SPOKEN_LANGUAGE]->(language:SpokenLanguage {name: 'English'})
#RETURN m1.name AS movie1, m3.name AS similar_movie, r.score AS similarity
#ORDER BY similarity DESCENDING, movie1, similar_movie LIMIT 10;
#"""
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
#    session.run(query_index)
#    vals = session.run(actors)

    results_list = [dict(record) for record in vals]
#    print(results_list)
    df = pd.DataFrame(results_list).drop_duplicates()       
#%%
query = "MATCH (n) with n limit 300 DETACH DELETE n;"
for i in range(1,1000):
    with driver.session() as session:
        session.run(query)
#%%
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
gds.run_cypher("CALL db.index.vector.createNodeIndex('overview-embeddings2', 'movie', 'textEmbedding', 768, 'cosine')")
gds.run_cypher(''' 
SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options
WHERE type = "VECTOR"
''')    
#%%
'''

The Below Code is for KNN


'''
# We define how we want to project our database into GDS
node_projection = ["Movie", "User"]
relationship_projection = {"RATING": {"orientation": "UNDIRECTED", "properties": "rating"}}

# Before actually going through with the projection, let's check how much memory is required
result = gds.graph.project.estimate(node_projection, relationship_projection)
#%%
# For this small graph memory requirement is low. Let us go through with the projection
G, result = gds.graph.project("movies", node_projection, relationship_projection)

print(f"The projection took {result['projectMillis']} ms")

# We can use convenience methods on `G` to check if the projection looks correct
print(f"Graph '{G.name()}' node count: {G.node_count()}")
print(f"Graph '{G.name()}' node labels: {G.node_labels()}")

#%%
# We can also estimate memory of running algorithms like FastRP, so let's do that first
result = gds.fastRP.mutate.estimate(
    G,
    mutateProperty="embedding",
    randomSeed=42,
    embeddingDimension=4,
    relationshipWeightProperty="rating",
    iterationWeights=[0.8, 1, 1, 1],
)

print(f"Required memory for running FastRP: {result['requiredMemory']}")
#%%
# Now let's run FastRP and mutate our projected graph 'purchases' with the results
result = gds.fastRP.mutate(
    G,
    mutateProperty="embedding",
    randomSeed=42,
    embeddingDimension=4,
    relationshipWeightProperty="rating",
    iterationWeights=[0.8, 1, 1, 1],
)

# Let's make sure we got an embedding for each node
print(f"Number of embedding vectors produced: {result['nodePropertiesWritten']}")
#%%

# Run kNN and write back to db (we skip memory estimation this time...)
result = gds.knn.write(
    G,
    topK=2,
    nodeProperties=["embedding"],
    randomSeed=42,
    concurrency=1,
    sampleRate=1.0,
    deltaThreshold=0.0,
    writeRelationshipType="SIMILAR",
    writeProperty="score",
)

print(f"Relationships produced: {result['relationshipsWritten']}")
print(f"Nodes compared: {result['nodesCompared']}")
print(f"Mean similarity: {result['similarityDistribution']['mean']}")
#%%
gds.run_cypher(
    """
        MATCH (p1:Person)-[r:SIMILAR]->(p2:Person)
        RETURN p1.name AS person1, p2.name AS person2, r.score AS similarity
        ORDER BY similarity DESCENDING, person1, person2
    """
)
#%%
'''

The Below Code is for GNN


'''
# Project a graph using the extended syntax
G, result = gds.graph.project(
    "extended-form-example-graph",
    {
        "Movie": {
            "label": "Movie","properties":{'overview':{'property':'overview', 'defaultValue':0}}
        },
        "User": {
            "label": "User","properties":{'id':{'property':'id', 'defaultValue':0}}
        },
        "Genre": {
            "label": "Genre","properties":{'name':{'property':'name', 'defaultValue':0}}          
        }
    },
    {
        "RATING": {"orientation": "UNDIRECTED", "properties": "rating"},
        "HAS_GENRE":{"orientation":"UNDIRECTED"}
    }
)
G.drop()
print(result)
#%%
# We define how we want to project our database into GDS

node_projection =     {
        "Movie": {
            "label": "Movie","properties":{'overview':{'property':'overview', 'defaultValue':0}}
        },
        "User": {
            "label": "User","properties":{'id':{'property':'id', 'defaultValue':0}}
        },
        "Genre": {
            "label": "Genre","properties":{'name':{'property':'name', 'defaultValue':0}}          
        }
    }
#node_projection = ["Movie", "User",'Genre']        
#node_projection = {'User': {'label': 'User', 'properties': {'id': {}}},
# 'Movie': {'label': 'Movie',

#  'properties': {'overview': {}}},
# 'Genre': {'label': 'Genre', 'properties': {'name': {}}}}
#node_projection = {
#        "Movie": {
#            "properties": ["overview"],'label':'Movie'
#        }}
#node_projection = {"Movie":{'properties':'overview'}}
relationship_projection = {
        "RATING": {"orientation": "UNDIRECTED", "properties": "rating"},
        "HAS_GENRE":{"orientation":"UNDIRECTED"}
        }

#node_properties = {'nodeProperties': ['overview','name']}
# Before actually going through with the projection, let's check how much memory is required
result = gds.graph.project.estimate(node_projection, relationship_projection)

# For this small graph memory requirement is low. Let us go through with the projection
G, result = gds.graph.project("movies",node_projection, relationship_projection)

print(f"The projection took {result['projectMillis']} ms")

# We can use convenience methods on `G` to check if the projection looks correct
print(f"Graph '{G.name()}' node count: {G.node_count()}")
print(f"Graph '{G.name()}' node labels: {G.node_labels()}")
print(f"Overview of G: {G}")
print(f"Node labels in G: {G.node_labels()}")
print(f"Relationship types in G: {G.relationship_types()}")
print(f"Node properties per label:\n{G.node_properties()}")
#%%

# Create an empty node classification pipeline
pipe, _ = gds.beta.pipeline.nodeClassification.create("genre-predictor-pipe")
#%%
# Set the test set size to be 79.6 % of the entire set of `Movie` nodes
_ = pipe.configureSplit(testFraction=0.796)
#%%

# Add a HashGNN node property step to the pipeline
_ = pipe.addNodeProperty(
    "beta.hashgnn",
    mutateProperty="embedding",
    iterations=4,
    heterogeneous=True,
    embeddingDensity=512,
    neighborInfluence=0.7,
    featureProperties=["overview"],
    randomSeed=41,
    contextNodeLabels=G.node_labels(),
)
#%%

# Set the embeddings vectors produced by HashGNN as feature input to our ML algorithm
_ = pipe.selectFeatures("embedding")
#%%
# Add logistic regression as a candidate ML algorithm for the training
# Provide an interval for the `penalty` parameter to enable autotuning for it
_ = pipe.addLogisticRegression(penalty=(0.1, 1.0), maxEpochs=1000, patience=5, tolerance=0.0001, learningRate=0.01)
#%%

# Add random forest as a candidate ML algorithm for the training
# Provide an interval for the `minSplitSize` parameter to enable autotuning for it
_ = pipe.addRandomForest(minSplitSize=(2, 100), criterion="ENTROPY")
#%%


# Call train on our pipeline object to run the entire training pipeline and produce a model
model, _ = pipe.train(
    G,
    modelName="genre-predictor-model",
    targetNodeLabels=["Movie"],
    targetProperty="overview",
    metrics=["F1_MACRO"],
    randomSeed=42,
)
     
#%%


print(f"Accuracy scores of trained model:\n{model.metrics()['F1_MACRO']}")
#%%

'''


Eigen Vector Centrality based on Above GRaph projection

'''


#eigenvector_centrality_result = gds.eigenvector.mutate(G, maxIterations=10000, mutateProperty="eigenvectorCentrality")
degree_centrality_result = gds.degree.mutate(G, mutateProperty="degreeCentrality")


# We can verify that the eigenvectorCentrality was mutated
G.node_properties()
#%%
if eigenvector_centrality_result.didConverge:
    print(
        f"The number of iterations taken by Eigenvector Centrality to run is {eigenvector_centrality_result.ranIterations}."
    )
else:
    print("Algorithm did not converge!")

#%%
import matplotlib.pyplot as plt    
    
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


#query = """
#MATCH (m:Movie) WHERE (m.year) IS NOT NULL
#WITH m.year AS year, count(*) AS count
#ORDER BY year
#RETURN toString(year) AS year, count
#"""
#
#gds.run_cypher(query)
#%%
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
del1 = "CALL gds.graph.drop('movies',false);"
del2 = "CALL gds.graph.drop('movies2',false);"
query1 = """CALL gds.graph.project('movies', 
              ['Movie','User','Genre'], 
              {RATING:{orientation: 'UNDIRECTED',properties:'rating'},HAS_GENRE:{orientation: 'UNDIRECTED'}}
            );"""

query2 = """CALL gds.fastRP.write(
          'movies',
          {
            nodeLabels:['Movie','Genre','User'],
            relationshipTypes:['RATING'],
            iterationWeights: [0.8,1.0],
            relationshipWeightProperty:'rating',
            embeddingDimension: 200,
            writeProperty: 'embedding_fastrp'
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
        RATING: {orientation: 'UNDIRECTED', properties: 'rating'},
        HAS_GENRE: {orientation: 'UNDIRECTED'}
    }
);

"""
query4 = """//Forming a KNN Graph
CALL gds.knn.write(
  'movies2',
  {
    nodeLabels:['Movie','User'] ,
    nodeProperties:'embedding_fastrp',
    topK: 2,
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
       COLLECT(DISTINCT movie_n.name) AS movies_rated_by_n,
       COLLECT(DISTINCT movie_m.name) AS movies_rated_by_m
LIMIT 100;

'''
for q in [del1,del2,query1,query2,query3,query4]:
    gds.run_cypher(q)
result = gds.run_cypher(verify)#graph.run(request).to_data_frame()


result['Common'] = result.apply(lambda x : list(set(x['movies_rated_by_n']).intersection(x['movies_rated_by_m'])),axis=1)
#https://community.neo4j.com/t/updating-in-memory-gds-projected-graph/63014/5
#%%


q = """MATCH (p:Person)-[r:ACTED_IN]->(m:Movie) where m.name = 'Titanic'
RETURN p.name,r.character,m.id limit 100;"""
#q = """MATCH (m:Movie) WHERE m.id = 9870
#RETURN m.name, m.id
#LIMIT 100;"""
a = gds.run_cypher(q)

q = "match (m:Movie) where m.name = 'Star Wars' return m.id"
gds.run_cypher(q)


q = "match (m:Person) where m.name = 'Tom Hanks' return m.name"
gds.run_cypher(q)


q = "matchh (m:Movie) where m.id in [597,2699,16535]"
gds.run_cypher(q)



q = "match (pc:SpokenLanguage) return pc.name limit 100"
gds.run_cypher(q)


q = """MATCH (pc:ProductionCompany)-[:PRODUCED_BY]->(m:Movie)
RETURN pc.name, COUNT(m) AS movieCount
LIMIT 100;
"""

q = "match (u:User) return u.userId limit 100"
gds.run_cypher(q)

#%%

wd2 = 'C:\\Users\\abhis\\.Neo4jDesktop\\relate-data\\dbmss\\dbms-270e66cc-c52d-46b5-9148-c8e81def25cf\\import\\\\'

df = pd.read_csv(wd2+"normalised_cast2.csv")

a = df[df['id']==9870]
b = df[(df['name']=='Tom Hanks')]

#%%
df = pd.read_csv(wd2+"normalised_crew.csv")

a = df.head(10)
b = df[(df['name']=='Tom Hanks')]
#%%
df = pd.read_csv(wd2+"normalised_production_countries.csv")

a = df.head(10)
b = df[(df['name']=='Tom Hanks')]


