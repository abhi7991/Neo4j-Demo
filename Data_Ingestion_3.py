# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 20:05:18 2024

@author: abhis
"""

import re
import os
import pandas as pd
wd = os.getcwd()
import random
import numpy as np
import base64
from neo4j import GraphDatabase
import os
from tqdm import tqdm
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

wd = os.getcwd()
np.random.seed(20)

limit2 = None #This is for Other nodes outside the class

def read_params_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return [line.strip() for line in lines]   

class createGraph:
    
    
    limit = None
    
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=200)

    def close(self):
        self.driver.close()

    def load_movies_from_csv(self, csv_file):
        
        '''
        
        Ensure the 'csv' file is in the import folder 
        linked to Neo4j
        
        '''
        print(f"LIMIT {self.limit}" if self.limit is not None else "")
        with self.driver.session() as session:
            # Cypher query to load movies from CSV
            query = (
                "LOAD CSV WITH HEADERS FROM $csvFile AS row "
                "WITH row " + (f"LIMIT {self.limit}" if self.limit is not None else "") +
                " CREATE (:Movie {name: row.original_title, "
                "id: toInteger(row.id), imdb_id: row.imdb_id, "
                "popularity: toFloat(coalesce(row.popularity, 0.0)), "
                "revenue: toInteger(coalesce(row.revenue, 0)), "
                "spoken_languages: row.spoken_languages, overview: row.overview, "
                "vote_average: toInteger(coalesce(row.vote_average, 0)), "
                "vote_count: toInteger(coalesce(row.vote_count, 0))});\n"
            )
           
            session.run(query, csvFile=f'file:///{csv_file}', limit=self.limit)
            print(f"Data Uploaded to Neo4j desktop for the first {self.limit} rows")

            query_create_index = "CREATE FULLTEXT INDEX movie_overview_index FOR (m:Movie) ON EACH [m.overview]"
            session.run(query_create_index)
            create_index = 'CREATE INDEX id_index FOR (m:Movie) ON (m.id)'
            session.run(create_index)
            
            print("Movie Overview Indexed")
            
            
    def drop_data(self):
        with self.driver.session() as session:
            # Check if data exists before attempting to delete
            check_query = "MATCH (n) RETURN count(n) AS count"
            result = session.run(check_query)
            count = result.single()["count"]

            if count > 0:
                # Data exists, proceed with dropping indices and deleting data
                try:
                    session.run("DROP INDEX movie_overview_index")
                except:
                    print("No index to drop")
                    
                try:
                    session.run("DROP INDEX id_index")
                except:
                    print("No index to drop")                    
                delete_data_query = (
                        "CALL apoc.periodic.iterate("
                        '"MATCH (n) RETURN n", "DETACH DELETE n", {batchSize: 10000});'
                    )
                session.run(delete_data_query)
                print("All indices dropped and data deleted")
            else:
                print("No data to delete")
                
    def search_movies_by_term(self, search_term):
        with self.driver.session() as session:
            # Cypher query to search for movies based on the full-text index
            query_search = (
                "CALL db.index.fulltext.queryNodes('movie_overview_index', $searchTerm) "
                "YIELD node,score "
                "RETURN node.name,node.overview, score"
            )
            result = session.run(query_search, searchTerm=search_term)
            # Convert the result to a list of dictionaries
            results_list = [dict(record) for record in result]

            return results_list 
    def load_users2(self):
        
        limit_clause = f"LIMIT {self.limit}" if self.limit is not None else ""
        with self.driver.session() as session:        
            session.run(
                f"CALL apoc.periodic.iterate("
                f"'LOAD CSV WITH HEADERS FROM \"file:///ratings_small.csv\" AS line RETURN line {limit_clause}', "
                "'MERGE (u:User { id: TOINTEGER(line.userId) }) "
                "SET u.userId = TOINTEGER(line.userId) "
                "WITH u, line "
                "MATCH (m:Movie { id: TOINTEGER(line.movieId) }) "
                "MERGE (u)-[r:RATING { rating: TOFLOAT(line.rating) }]->(m) "
                "RETURN COUNT(*) AS processedRows', "
                "{ batchSize: 1000});"
            )
        print("Users Uploaded")        
    def loadNodes(self):    
        parameters = {"limit": f"{self.limit}" if self.limit is not None else ""}
        create_prod_comp_query = """CALL apoc.periodic.iterate(
              'LOAD CSV WITH HEADERS FROM "file:///normalised_production_companies.csv" AS row RETURN row', 
              'MERGE (pc:ProductionCompany { id: row.production_companies_id}) ' +
              'ON CREATE SET pc.name = row.name ' +
              'WITH pc, row ' +
              'MATCH (m:Movie { id: TOINTEGER(row.id) }) ' +
              'MERGE (m)-[:PRODUCED_BY]->(pc) ' +
              'ON CREATE SET pc.productionMovieId = m.id', 
              { batchSize: 100,parallel:true, concurrency:10}
            ) YIELD batches, total, errorMessages;"""

        create_genres_query = """CALL apoc.periodic.iterate(
            'LOAD CSV WITH HEADERS FROM "file:///normalised_genres.csv" AS row RETURN row', 
            'MERGE (g:Genre { id: row.genres_id}) ' +
            'ON CREATE SET g.name = row.name ' +
            'WITH g, row ' +
            'MATCH (m:Movie { id: TOINTEGER(row.id) }) ' +
            'MERGE (m)-[:HAS_GENRE]->(g)', 
            { batchSize: 100, parallel:true, concurrency:10 }
        ) YIELD batches, total, errorMessages;"""
        
        create_spoken_languages_query = """CALL apoc.periodic.iterate(
            'LOAD CSV WITH HEADERS FROM "file:///normalised_spoken_languages.csv" AS row RETURN row', 
            'MERGE (sl:SpokenLanguage { id: row.spoken_languages_id}) ' +
            'ON CREATE SET sl.name = row.name ' +
            'WITH sl, row ' +
            'MATCH (m:Movie { id: TOINTEGER(row.id) }) ' +
            'MERGE (m)-[:HAS_SPOKEN_LANGUAGE]->(sl)', 
            { batchSize: 100, parallel:true, concurrency:10 }
        ) YIELD batches, total, errorMessages;"""
        
        create_prod_countries_query = """CALL apoc.periodic.iterate(
            'LOAD CSV WITH HEADERS FROM "file:///normalised_production_countries.csv" AS row RETURN row', 
            'MERGE (pcn:ProductionCountry { id: row.production_countries_id}) ' +
            'ON CREATE SET pcn.name = row.name ' +
            'WITH pcn, row ' +
            'MATCH (m:Movie { id: TOINTEGER(row.id) }) ' +
            'MERGE (m)-[:PRODUCED_IN]->(pcn)', 
            { batchSize: 100, parallel:true, concurrency:10 }
        ) YIELD batches, total, errorMessages;"""

        queries = [
            create_prod_comp_query,
            create_genres_query,
            create_spoken_languages_query,
            create_prod_countries_query
        ]               
        with self.driver.session() as session:        
#            session.run(create_prod_comp_query,parameters)
#            session.run(create_relationships_query,parameters)   
            for query in queries:
                session.run(query, parameters)      
            print("All Done with movies")
            

    def actors(self):
        parameters = {"limit": f"{self.limit}" if self.limit is not None else ""}
        
        create_cast_query = """CALL apoc.periodic.iterate(
            'LOAD CSV WITH HEADERS FROM "file:///normalised_cast.csv" AS row RETURN row', 
            'MERGE (p:Person { id: row.cast_id}) ' +
            'ON CREATE SET p.name = row.name, p.gender = row.gender, p.profile_path = row.profile_path ' +
            'WITH p, row ' +
            'MATCH (m:Movie { id: TOINTEGER(row.id) }) ' +
            'MERGE (m)<-[:ACTED_IN { character: row.character, credit_id: row.credit_id, order: row.order }]-(p)', 
            { batchSize: 100}
        ) YIELD batches, total, errorMessages;"""
        
        with self.driver.session() as session:        
            session.run(create_cast_query, parameters)      
            print("All Done with actors")            

    def crew(self):
        parameters = {"limit": f"{self.limit}" if self.limit is not None else ""}

        create_crew_query = """CALL apoc.periodic.iterate(
            'LOAD CSV WITH HEADERS FROM "file:///normalised_crew.csv" AS row RETURN row', 
            'MERGE (p:Person { id: row.credit_id }) ' +
            'ON CREATE SET p.name = row.name, p.gender = row.gender, p.profile_path = row.profile_path ' +
            'WITH p, row ' +
            'MATCH (m:Movie { id: TOINTEGER(row.id) }) ' +
            'MERGE (m)<-[:CREWED_IN { job: row.job }]-(p)', 
            { batchSize: 100, parallel:true, concurrency:10 }
        ) YIELD batches, total, errorMessages;"""
        with self.driver.session() as session:        
            session.run(create_crew_query, parameters)      
            print("All Done with Crew")            

    def load_overview_embeddings(self):
        
        with open('movie_embeddings.pickle', 'rb') as f:
                embedding_dict = pickle.load(f)
        
        with self.driver.session() as session:
            for idx, (id, embedding) in tqdm(enumerate(embedding_dict.items())):
                # Set your desired limit, for example, 100
                if idx >= self.limit:
                    break
                query = (
                    "MATCH (m:Movie {id: toInteger($movieId)}) "
                    "CALL db.create.setNodeVectorProperty(m, 'embedding', $embedding) "
                )                 
                session.run(query, movieId=id, embedding=embedding) 
            
            print("Overview Embeddings loaded to Neo4j desktop")
            
            try:
                session.run("DROP INDEX overview_embeddings")
            except:
                print("No index to drop")
        
            query_index = (
                        "CREATE VECTOR INDEX overview_embeddings "
                        "FOR (m: Movie) ON (m.embedding) "
                        "OPTIONS {indexConfig: { "
                        "`vector.dimensions`: 768, "
                        "`vector.similarity_function`: 'cosine'}}"
                    )            
                    
            session.run(query_index)   
            print("Overview Vector index created in Neo4j desktop")
            
'Read the credentials to your database'
start = time.time()
#Check Limits below and above
uri, user, password = read_params_from_file(wd+"\\params.txt") 

movieGraph = createGraph(uri, user, password)
del password
 
reCreate = True
others = True
actors = True
crew = True 
embeddings = False
if reCreate:

    movieGraph.drop_data()
    movieGraph.load_movies_from_csv("movies_metadata.csv")#Linked to Import Folder of neo4j
    movieGraph.load_users2()            
    
if others:
   movieGraph.loadNodes()    

if actors:
    movieGraph.actors()
    
if crew:
    movieGraph.crew()    
    
if embeddings:
    movieGraph.load_overview_embeddings()    
    
end = time.time()
print("Elapsed Time : ", end - start)
#   
#modify the queries based on the conditions below and change the relationships as well? 

#['id', 'production_companies_id', 'name'] - normalised_production_companies.csv
#['id', 'genres_id', 'name'], dtype='object') -  normalised_genres.csv
#['id', 'spoken_languages_id', 'name'] - normalised_spoken_languages.csv
#['id', 'production_countries_id', 'name'] - normalised_production_countries