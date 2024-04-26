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
from dotenv import load_dotenv
warnings.filterwarnings("ignore")

load_dotenv()

wd = os.getcwd()
np.random.seed(20)

limit2 = 1000 #This is for Other nodes outside the class

def read_params_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return [line.strip() for line in lines]   

class createGraph:
    
    

    
    def __init__(self, uri, user, password):
        # database = 'movies.main'
        self.driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=200)

    def close(self):
        self.driver.close()

    def createConstraint(self):
        query1 = "CREATE CONSTRAINT unique_movie_id IF NOT EXISTS FOR (p:Movie) REQUIRE (p.id) IS NODE KEY;"
        query2 = "CREATE CONSTRAINT unique_person_id IF NOT EXISTS FOR (p:Person) REQUIRE (p.person_id) IS NODE KEY;"
        query3 = "CREATE CONSTRAINT unique_prod_id IF NOT EXISTS FOR (p:ProductionCompany) REQUIRE (p.prod_comp_id) IS NODE KEY;"
        query4 = "CREATE CONSTRAINT unique_genre_id IF NOT EXISTS FOR (p:Genre) REQUIRE (p.genre_id) IS NODE KEY;"
        query5 = "CREATE CONSTRAINT unique_lang_id IF NOT EXISTS FOR (p:SpokenLanguage) REQUIRE (p.lang_id) IS NODE KEY;"
        query6 = "CREATE CONSTRAINT unique_countries_id IF NOT EXISTS FOR (p:Country) REQUIRE (p.country_id) IS NODE KEY;"
        query7 = "CREATE CONSTRAINT unique_user_id IF NOT EXISTS FOR (p:User) REQUIRE (p.userId) IS NODE KEY;"

        with self.driver.session() as session:
            session.run(query1)
            session.run(query2)
            session.run(query3)
            session.run(query4)
            session.run(query5)
            session.run(query6)
            session.run(query7)
    def load_movies_from_csv(self, csv_file):
        
        '''
        
        Ensure the 'csv' file is in the import folder 
        linked to Neo4j
        
        '''
        print(f"LIMIT {self.limit}" if self.limit is not None else "No LIMIT set")
        with self.driver.session() as session:
            # Cypher query to load movies from CSV
#            query = (
#                "LOAD CSV WITH HEADERS FROM $csvFile AS row "
#                "WITH row " + (f"LIMIT {self.limit}" if self.limit is not None else "") +
#                " CREATE (:Movie {name: row.original_title, "
#                "id: toInteger(row.id), imdb_id: row.imdb_id, "
#                "popularity: toFloat(coalesce(row.popularity, 0.0)), "
#                "revenue: toInteger(coalesce(row.revenue, 0)), "
#                "spoken_languages: row.spoken_languages, overview: row.overview, "
#                "vote_average: toInteger(coalesce(row.vote_average, 0)), "
#                "vote_count: toInteger(coalesce(row.vote_count, 0))});\n"
#            )
            query = (
                "LOAD CSV WITH HEADERS FROM $csvFile AS row "
                "WITH row WHERE row.id IS NOT NULL "  # Filter out rows with null id
                + (f"LIMIT {self.limit} " if self.limit is not None else "") +
                "MERGE (m:Movie {id: toInteger(row.id)}) "
                "ON CREATE SET m.name = row.original_title, "
                "m.imdb_id = row.imdb_id, "
                "m.popularity = toFloat(coalesce(row.popularity, 0.0)), "
                "m.revenue = toInteger(coalesce(row.revenue, 0)), "
                "m.spoken_languages = row.spoken_languages, "
                "m.adult = toFloat(coalesce(row.adult, 0.0)),"
                "m.budget = toFloat(coalesce(row.budget,0.0)),"
                "m.overview = row.overview, "
                "m.vote_average = toInteger(coalesce(row.vote_average, 0)), "
                "m.vote_count = toInteger(coalesce(row.vote_count, 0)),"
                "m.embedding = '';\n"
            )



            session.run(query, csvFile=f'file:///{csv_file}', limit=self.limit)
            print(f"Data Uploaded to Neo4j desktop for the first {self.limit} rows")

            
    limit = None            
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
                    
                try: 
                    session.run("DROP CONSTRAINT unique_movie_id IF EXISTS;")
                except:
                    print("No constraint to drop : unique_movie_id")

                try: 
                    session.run("DROP CONSTRAINT unique_person_id IF EXISTS;")
                except:
                    print("No constraint to drop : unique_person_id")  

                try: 
                    session.run("DROP CONSTRAINT unique_prod_id IF EXISTS;")
                except:
                    print("No constraint to drop : unique_prod_id")  

                try: 
                    session.run("DROP CONSTRAINT unique_genre_id IF EXISTS;")
                except:
                    print("No constraint to drop : unique_genre_id")  

                try: 
                    session.run("DROP CONSTRAINT unique_lang_id IF EXISTS;")
                except:
                    print("No constraint to drop : unique_lang_id")  

                try: 
                    session.run("DROP CONSTRAINT unique_countries_id IF EXISTS;")
                except:
                    print("No constraint to drop : unique_countries_id")                      

                try: 
                    session.run("DROP CONSTRAINT unique_user_id IF EXISTS;")
                except:
                    print("No constraint to drop : unique_user_id")  

#                delete_data_query = (
#                        "CALL apoc.periodic.iterate("
#                        '"MATCH (n) RETURN n", "DETACH DELETE n", {batchSize: 10000});'
#                    )
#                session.run(delete_data_query)
                try:    
                    query = "MATCH (n) with n limit 30000 DETACH DELETE n;"
                    for i in range(1,100):
                        session.run(query)              
                except:
                    query = """CALL apoc.periodic.iterate(
                        "MATCH (n) RETURN n", "DETACH DELETE n", {batchSize: 100});"""
                    session.run(query)
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

                
            create_user_node = """CALL apoc.periodic.iterate(
                    'LOAD CSV WITH HEADERS FROM "file:///clean_ratings.csv" AS row RETURN row', 
                    'MERGE (pc:User {userId: TOINTEGER(row.userId)})', 
                    { batchSize: 100}
                ) YIELD batches, total, errorMessages;
                """
            create_user_relay = """CALL apoc.periodic.iterate(
                    'LOAD CSV WITH HEADERS FROM "file:///clean_ratings.csv" AS row RETURN row', 
                        'MATCH (m:Movie {id: TOINTEGER(row.movieId)}) ' +
                        'MATCH (pc:User {userId: TOINTEGER(row.userId)}) ' +
                        'MERGE (pc)-[r:RATING { rating: TOFLOAT(row.rating) }]->(m) ', 
                        { batchSize: 100}
                    ) YIELD batches, total, errorMessages;
                    """   
            session.run(create_user_node)        
            session.run(create_user_relay)
        print("Users Uploaded")        
    def loadNodes(self):    
        parameters = {"limit": f"{self.limit}" if self.limit is not None else ""}

        create_prod_com_nodes = """CALL apoc.periodic.iterate(
                'LOAD CSV WITH HEADERS FROM "file:///normalised_production_companies.csv" AS row RETURN row', 
                'MERGE (pc:ProductionCompany {prod_comp_id: TOINTEGER(row.production_companies_id)}) ' +
                'ON CREATE SET pc.name = row.name', 
                { batchSize: 100}
            ) YIELD batches, total, errorMessages;
            """
        create_prod_comp_relay = """CALL apoc.periodic.iterate(
                'LOAD CSV WITH HEADERS FROM "file:///normalised_production_companies.csv" AS row RETURN row', 
                    'MATCH (m:Movie {id: TOINTEGER(row.id) }) ' +
                    'MATCH (pc:ProductionCompany {prod_comp_id: TOINTEGER(row.production_companies_id)}) ' +
                    'MERGE (pc)-[:PRODUCED_BY]->(m) ', 
                    { batchSize: 100}
                ) YIELD batches, total, errorMessages;
                """
                
        create_genres_node = """CALL apoc.periodic.iterate(
                'LOAD CSV WITH HEADERS FROM "file:///normalised_genres.csv" AS row RETURN row', 
                'MERGE (pc:Genre {genre_id: TOINTEGER(row.genres_id)}) ' +
                'ON CREATE SET pc.name = row.name', 
                { batchSize: 100}
            ) YIELD batches, total, errorMessages;
            """
        create_genres_relay = """CALL apoc.periodic.iterate(
                'LOAD CSV WITH HEADERS FROM "file:///normalised_genres.csv" AS row RETURN row', 
                    'MATCH (m:Movie { id: TOINTEGER(row.id) }) ' +
                    'MATCH (pc:Genre { genre_id: TOINTEGER(row.genres_id)}) ' +
                    'MERGE (pc)-[:GENRE]->(m) ', 
                    { batchSize: 100}
                ) YIELD batches, total, errorMessages;
                """

        create_lang_node = """CALL apoc.periodic.iterate(
                'LOAD CSV WITH HEADERS FROM "file:///normalised_spoken_languages.csv" AS row RETURN row', 
                'MERGE (pc:SpokenLanguage {lang_id: row.spoken_languages_id}) ' +
                'ON CREATE SET pc.name = row.name', 
                { batchSize: 100}
            ) YIELD batches, total, errorMessages;
            """
        create_lang_relay = """CALL apoc.periodic.iterate(
                'LOAD CSV WITH HEADERS FROM "file:///normalised_spoken_languages.csv" AS row RETURN row', 
                    'MATCH (m:Movie {id: TOINTEGER(row.id)}) ' +
                    'MATCH (pc:SpokenLanguage {lang_id: row.spoken_languages_id}) ' +
                    'MERGE (pc)-[:LANGUAGE]->(m) ', 
                    { batchSize: 100}
                ) YIELD batches, total, errorMessages;
                """        

        create_country_node = """CALL apoc.periodic.iterate(
                'LOAD CSV WITH HEADERS FROM "file:///normalised_production_countries.csv" AS row RETURN row', 
                'MERGE (pc:Country {country_id: row.production_countries_id}) ' +
                'ON CREATE SET pc.name = row.name', 
                { batchSize: 100}
            ) YIELD batches, total, errorMessages;
            """
        create_country_relay = """CALL apoc.periodic.iterate(
                'LOAD CSV WITH HEADERS FROM "file:///normalised_production_countries.csv" AS row RETURN row', 
                    'MATCH (m:Movie {id: TOINTEGER(row.id)}) ' +
                    'MATCH (pc:Country {country_id: row.production_countries_id}) ' +
                    'MERGE (pc)-[:COUNTRY]->(m) ', 
                    { batchSize: 100}
                ) YIELD batches, total, errorMessages;
                """   

        create_year_nodes = """
                CALL apoc.periodic.iterate(
                    'LOAD CSV WITH HEADERS FROM "file:///keywords_clean.csv" AS row 
                     WITH row WHERE row.release_year IS NOT NULL
                     RETURN row', 
                    'MERGE (year:Year {year: TOFLOAT(row.release_year)})',
                    { batchSize: 100 }
                ) YIELD batches, total, errorMessages
                RETURN batches, total, errorMessages
                """
        create_year_relay = """CALL apoc.periodic.iterate(
                'LOAD CSV WITH HEADERS FROM "file:///keywords_clean.csv" AS row RETURN row', 
                'WITH row WHERE row.release_year IS NOT NULL ' +
                'MATCH (m:Movie {id: TOINTEGER(row.id)}) ' +
                'MATCH (year:Year {year: TOFLOAT(row.release_year)}) ' +
                'MERGE (year)-[:RELEASED]->(m)',
                { batchSize: 100}
            ) YIELD batches, total, errorMessages;
                """
        queries = [
                create_prod_com_nodes,create_prod_comp_relay,create_genres_node,create_genres_relay,
                create_lang_node,create_lang_relay,create_country_node,create_country_relay,create_year_nodes,create_year_relay
                
        ]                
        with self.driver.session() as session:        
            for query in queries:
                session.run(query, parameters)      
            print("All Done with movies")
            

    def actors(self):
        parameters = {"limit": f"{self.limit}" if self.limit is not None else ""}
        

        create_cast_query = """CALL apoc.periodic.iterate(
                'LOAD CSV WITH HEADERS FROM "file:///normalised_cast2.csv" AS row RETURN row', 
                'MERGE (p:Person {person_id: TOINTEGER(row.r_id)}) ' +
                'ON CREATE SET p.name = row.name, p.gender = TOINTEGER(row.gender), p.profile_path = row.profile_path', 
                { batchSize: 100 }
            ) YIELD batches, total, errorMessages;
                    """
        create_cast_relay = """CALL apoc.periodic.iterate(
                'LOAD CSV WITH HEADERS FROM "file:///normalised_cast2.csv" AS row RETURN row', 
                'MATCH (m:Movie {id: TOINTEGER(row.id) }) ' +
                'MATCH (p:Person {person_id:TOINTEGER(row.r_id) }) ' +
                'MERGE (p)-[:ACTED_IN { character: row.character, credit_id: row.credit_id, `order`: row.order }]->(m)', 
                { batchSize: 100 }
            ) YIELD batches, total, errorMessages;
                        """
        with self.driver.session() as session:        
            session.run(create_cast_query, parameters)      
            session.run(create_cast_relay, parameters)    
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
        
        
        create_crew_query = """CALL apoc.periodic.iterate(
                'LOAD CSV WITH HEADERS FROM "file:///normalised_crew.csv" AS row RETURN row', 
                'MERGE (p:Person {person_id: TOINTEGER(row.r_id)}) ' +
                'ON CREATE SET p.name = row.name, p.gender = row.gender', 
                { batchSize: 100 }
            ) YIELD batches, total, errorMessages;
                    """
        create_crew_relay = """CALL apoc.periodic.iterate(
                'LOAD CSV WITH HEADERS FROM "file:///normalised_crew.csv" AS row RETURN row', 
                'MATCH (m:Movie {id: TOINTEGER(row.id) }) ' +
                'MATCH (p:Person {person_id:TOINTEGER(row.r_id) }) ' +
                'MERGE (p)-[:CREWED_IN { character: row.department, credit_id: row.credit_id}]->(m)', 
                { batchSize: 100 }
            ) YIELD batches, total, errorMessages;
                        """        
        
        with self.driver.session() as session:        
            session.run(create_crew_query, parameters)  
            session.run(create_crew_relay, parameters)
            print("All Done with Crew")            

    def load_overview_embeddings(self):
        
        with open('movie_embeddings.pickle', 'rb') as f:
                embedding_dict = pickle.load(f)
        
        with self.driver.session() as session:
            for idx, (id, embedding) in tqdm(enumerate(embedding_dict.items())):
                # Set your desired limit, for example, 100
#                idx = {False} if self.limit is not None else true)
                idx1 = {False} if self.limit is not None else True
                if idx1:            
                    if idx >= self.limit:
                        break
                query = (
                    "MATCH (m:Movie {id: toInteger($movieId)}) "
                    "CALL db.create.setNodeVectorProperty(m, 'embedding', $embedding) "
                )                 
                session.run(query, movieId=id, embedding=embedding) 
            
            print("Overview Embeddings loaded to Neo4j desktop")
            
#            try:
#                session.run("DROP INDEX overview_embeddings")
#            except:
#                print("No index to drop")
#        
#            query_index = (
#                        "CREATE VECTOR INDEX overview_embeddings "
#                        "FOR (m: Movie) ON (m.embedding) "
#                        "OPTIONS {indexConfig: { "
#                        "`vector.dimensions`: 768, "
#                        "`vector.similarity_function`: 'cosine'}}"
#                    )            
#                    
#            session.run(query_index)   
#            print("Overview Vector index created in Neo4j desktop")

    def loadEmbeddings(self):
        with open('movie_embeddings.pickle', 'rb') as f:
            embedding_dict = pickle.load(f)
        
        # Batch size for processing embeddings
        batch_size = 1000
        
        with self.driver.session() as session:
            # Prepare the Cypher query template
            query_template = (
                "UNWIND $batch AS item "
                "MATCH (m:Movie {id: toInteger(item.id)}) "
                "SET m.embedding = item.embedding"
            )
        
            # Process embeddings in batches
            for i in range(0, len(embedding_dict), batch_size):
                # Slice the embeddings dictionary to get a batch
                batch = [{"id": id, "embedding": embedding} for id, embedding in embedding_dict.items()][i:i+batch_size]
        
                # Execute the query with the current batch
                session.run(query_template, batch=batch)
        
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
    
    def loadKeyword(self):
        with self.driver.session() as session:
            # Prepare the Cypher query template
            batch = 1000            
            query_template = (
                """CALL apoc.periodic.iterate(
                'LOAD CSV WITH HEADERS FROM "file:///keywords_clean.csv" AS row RETURN row', 
                'MATCH (m:Movie {id: toFloat(toInteger(row.id))}) ' + 
                'SET m.keyword = row.Keywords2, m.collection = row.collection',
                { batchSize: 100 }
            ) YIELD batches, total, errorMessages;"""
            )    
            session.run(query_template, batch=batch)
            print("Keywords added")

            
# 'Read the credentials to your database'


def create_movie_graph():
    start = time.time()
    uri, user, password = os.getenv('NEO4J_URI'), os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD')
    movieGraph = createGraph(uri, user, password)
    del password
    
    reCreate = True
    users = True
    others = True
    actors = True
    crew = True 
    embeddings = False
    keywords = True  
    if reCreate:

        movieGraph.drop_data()
        movieGraph.createConstraint()
        movieGraph.load_movies_from_csv("movies_metadata_clean_2.csv")#Linked to Import Folder of neo4j
                
    if others:
        movieGraph.loadNodes()    
    if actors:
        movieGraph.actors()    
    if crew:
        movieGraph.crew()        
    if embeddings:
    #    movieGraph.load_overview_embeddings()    
        movieGraph.loadEmbeddings()
    if keywords:
        movieGraph.loadKeyword()
    if users:
        movieGraph.load_users2()           
    end = time.time()
    print("Elapsed Time : ", end - start)