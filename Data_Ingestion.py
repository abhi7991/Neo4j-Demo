# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:03:02 2024

@author: abhis
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 20:39:08 2024

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
import time
import warnings
warnings.filterwarnings("ignore")

wd = os.getcwd()
np.random.seed(20)



def read_params_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return [line.strip() for line in lines]   

class createGraph:
    
    
    limit = 10
    
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

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
            # Create index after loading data
            #            query_create_index = "CREATE INDEX FOR (m:Movie) ON (m.overview)"
            query_create_index = "CREATE FULLTEXT INDEX movie_overview_index FOR (m:Movie) ON EACH [m.overview]"
            session.run(query_create_index)
           
            print("Movie Overview Indexed")
            
            
    def drop_data(self):
        with self.driver.session() as session:
            # Check if data exists before attempting to delete
            check_query = "MATCH (n) RETURN count(n) AS count"
            result = session.run(check_query)
            count = result.single()["count"]

            if count > 0:
                # Data exists, proceed with dropping indices and deleting data
                session.run("DROP INDEX movie_overview_index")
                delete_data_query = "MATCH (n) DETACH DELETE n;"
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
                
    def loadNodes(self, row):
        # Read the CSV file into a pandas DataFrame
#        df = pd.read_csv(csv_file).head(self.limit)

        with self.driver.session() as session:
            # Iterate over rows in the DataFrame
#            for index, row in df.iterrows():
            try:
                '''
    
                Production Companies
    
                '''
                col = 'production_companies' 
                # Clean the production_companies column using ast.literal_eval
                production_companies_list = eval(row[col]) if isinstance(eval(row[col]), str) else eval(row[col])#eval(row['production_companies'])
                
                # Cypher query to load production companies from DataFrame
    
                for company in production_companies_list:
                    query = (
                        "MERGE (m:Movie {id: toInteger($movieId)}) "
                        "WITH m UNWIND $companies AS company "
                        "MERGE (p:ProductionCompany {name: company.name}) ON CREATE SET p.id = toInteger(company.id) "
                        "MERGE (m)-[:PRODUCED_BY]->(p)"
                    )                
                    session.run(query, movieId=row['id'], companies=company)
            except:
                pass
#            print("LOADING HAPPENING")
            '''

            Genres

            '''

#            genres = eval(row['genres'])
            try:
                col = 'genres'
                genres = eval(row[col]) if isinstance(eval(row[col]), str) else eval(row[col])#eval(eval(row['genres'])) if isinstance(eval(row['genres'])
                for genre in genres:
                    query = (
                        "MERGE (m:Movie {id: toInteger($movieId)}) "
                        "WITH m UNWIND $genres AS genre "
                        "MERGE (p:Genre {name: genre.name}) ON CREATE SET p.id = toInteger(genre.id) "
                        "MERGE (m)-[:GENRE]->(p)"
                    )                
    
                    session.run(query, movieId=row['id'], genres=genre)
            except:
                pass

            '''

            Languages

            '''
            try:
                # Clean the production_companies column using ast.literal_eval
                col ='spoken_languages'
                languages =  eval(row[col]) if isinstance(eval(row[col]), str) else eval(row[col])#eval(row['spoken_languages'])
    
                # Cypher query to load production companies from DataFrame
    
                for lang in languages:
                    query = (
                        "MERGE (m:Movie {id: toInteger($movieId)}) "
                        "WITH m UNWIND $languages AS lang "
                        "MERGE (p:Language {name: lang.name}) ON CREATE SET p.id = toInteger(lang.id) "
                        "MERGE (m)-[:LANGUAGES]->(p)"
                    )                
    
                    session.run(query, movieId=row['id'], languages=lang)               
            except:
                pass

            '''
            
            Countries

            ''' 
            try:
                # Clean the production_companies column using ast.literal_eval
                col = "production_countries"
                countries =  eval(row[col]) if isinstance(eval(row[col]), str) else eval(row[col])#eval(row['production_countries'])
    
                # Cypher query to load production companies from DataFrame
    
                for country in countries:
                    query = (
                        "MERGE (m:Movie {id: toInteger($movieId)}) "
                        "WITH m UNWIND $countries AS country "
                        "MERGE (p:Country {name: country.name}) ON CREATE SET p.id = toInteger(country.id) "
                        "MERGE (m)-[:LANGUAGES]->(p)"
                    )                
    
                    session.run(query, movieId=row['id'], countries=country)                
            except:
                pass

    def load_actors(self,file):
        df = pd.read_csv(file)
        df = df.head(self.limit)
        for i,r in df.iterrows():
            cast = pd.DataFrame(eval(df.iloc[:,0][i]))
            cast['movie_id'] = df.iloc[:,2][i]
        
            for _, row in cast.iterrows():
                actor_id = row['id']
                name = row['name']
                gender = row['gender']
                movie_id = row['movie_id']
                character = row['character']
                order = row['order']       
                
                movieGraph.create_actor_node(actor_id, name, gender)
                movieGraph.connect_actor_to_movie(actor_id, movie_id, character, order)
        print("Actors Data uploaded to Neo4j desktop and relationships created")                    

    def create_actor_node(self, actor_id, name, gender):
        with self.driver.session() as session:
            session.run(
                "MERGE (a:Actor {id: $id}) "
                "SET a.name = $name, a.gender = $gender",
                id=actor_id,
                name=name,
                gender=gender
            )

    def connect_actor_to_movie(self, actor_id, movie_id, character, order):
        with self.driver.session() as session:
            session.run(
                "MATCH (a:Actor {id: $actor_id}) "
                "MATCH (m:Movie {id: $movie_id}) "
                "MERGE (a)-[:ACTED_IN {character: $character, order: $order}]->(m)",
                actor_id=actor_id,
                movie_id=movie_id,
                character=character,
                order=order
            )        
            
    def load_crew(self, file):
        df = pd.read_csv(file).head(self.limit)
        for i,r in df.iterrows():
            cast = pd.DataFrame(eval(df.iloc[:,1][i]))
            cast['movie_id'] = df.iloc[:,2][i]        
            for i, r in cast.iterrows():
                person_id = r['id']
                name = r['name']
                gender = r['gender']
                department = r['department']
                job = r['job']
                movie_id = r['movie_id'] 
                if (("lighting" in department.lower()) |("costume" in department.lower()) | ("makeup" in department.lower()) | ("camera" in department.lower())):
                    pass
                else:
                    movieGraph.create_crew_node(person_id, name, gender, department, job)
                    movieGraph.connect_crew_to_movie(person_id, movie_id, department)
        
#            print("Crew Data uploaded to Neo4j desktop and relationships created")                    
    
    def create_crew_node(self, person_id, name, gender, department, job):
        dept = re.sub(r'[^\w\s]', '', department).replace(' ', '_').lower()        
        with self.driver.session() as session:
            session.run(
                "MERGE (p:"+dept.replace(" ","_")+" {id: $id}) "
                "SET p.name = $name, p.gender = $gender, p.department = $department, p.job = $job",
                id=person_id,
                name=name,
                gender=gender,
                department=department,
                job=job
            )
    
    def connect_crew_to_movie(self, person_id, movie_id, department):
        dept = re.sub(r'[^\w\s]', '', department).replace(' ', '_').lower()

        with self.driver.session() as session:
            session.run(
                "MATCH (p:"+dept.replace(" ","_")+" {id: $person_id}) "
                "MATCH (m:Movie {id: $movie_id}) "
                "MERGE (p)-[:"+dept.upper()+"]->(m)",
                person_id=person_id,
                movie_id=movie_id,
                department=department
            )

#%%
            
'Read the credentials to your database'
start = time.time()
#Check Limits below and above
uri, user, password = read_params_from_file(wd+"\\params.txt") 

movieGraph = createGraph(uri, user, password)
del password
movieGraph.drop_data()

movieGraph.load_movies_from_csv("movies_metadata.csv")#Linked to Import Folder of neo4j


others = True
if others:
    df = pd.read_csv(wd+"\\movies_metadata.csv")
    df.head().apply(lambda x  : movieGraph.loadNodes(x), axis = 1)


actors = True
if actors:
    movieGraph.load_actors(wd+"\\credits.csv")

crew = True

if crew:
    movieGraph.load_crew(wd+"\\credits.csv")


end = time.time()
print("Elapsed Time : ", end - start)

#