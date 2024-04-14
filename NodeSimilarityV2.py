# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:16:27 2024

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
from dotenv import load_dotenv
load_dotenv()


database = os.environ.get('NEO4J_DATABASE')
driver = GraphDatabase.driver(os.environ.get('NEO4J_URI'), auth=(os.environ.get('NEO4J_USERNAME'), os.environ.get('NEO4J_PASSWORD')), max_connection_lifetime=200, database = database)
gds = GraphDataScience(
    os.environ.get('NEO4J_URI'),
    auth = (os.environ.get('NEO4J_USERNAME'),os.environ.get('NEO4J_PASSWORD')),database = database
)       

del1 = "CALL gds.graph.drop('movies2');"
del2 = "CALL gds.graph.drop('movies3');"


gds.run_cypher(del1)
gds.run_cypher(del2)

query1 = """CALL gds.graph.project('movies2', 
              ['Movie','Genre','User','Person','SpokenLanguage','ProductionCompany','Country'], 
              {RATING:{properties:'rating',orientation:'REVERSE'},
              GENRE:{orientation:'REVERSE'},
              ACTED_IN:{orientation:'REVERSE'},
              CREWED_IN:{orientation:'REVERSE'},
              LANGUAGE:{orientation:'REVERSE'},
              PRODUCED_BY:{orientation:'REVERSE'},
              COUNTRY:{orientation:'REVERSE'}}   
            );"""


a = gds.run_cypher(query1)
print(a)

query2 = """CALL gds.graph.project('movies3', 
              ['Movie','Genre','User','Person','ProductionCompany'], 
              {RATING:{properties:'rating'},
              GENRE:{},
              ACTED_IN:{},
              CREWED_IN:{}, PRODUCED_BY:{}}  
            );"""
a = gds.run_cypher(query2)
print(a)

def getSimilar(entity,sim_type=''):
    
    
    #Similar movies based on user rating
    if sim_type.lower()=='user':
        node = """
        MATCH (m:entity)
        WHERE tolower(m.name) = '"""+entity.lower()+"""'
        WITH id(m) AS sourceNodeId
        CALL gds.nodeSimilarity.filtered.stream('movies2',{
            nodeLabels:['Movie','User','SpokenLanguage'],
            relationshipTypes:['RATING','LANGUAGE'],
            sourceNodeFilter: sourceNodeId,
            targetNodeFilter:'Movie'
        })
        YIELD node1, node2, similarity
        RETURN gds.util.asNode(node1).name AS Person1, gds.util.asNode(node2).name AS Person2, similarity
        ORDER BY similarity DESCENDING, Person1, Person2
        """        
    #Similar movies based on Genre
    elif sim_type.lower()=='genre':
        node = """
        MATCH (m:Movie)
        WHERE tolower(m.name) = '"""+entity.lower()+"""'
        WITH id(m) AS sourceNodeId
        CALL gds.nodeSimilarity.filtered.stream('movies2',{
            nodeLabels:['Movie','Genre'],
            relationshipTypes:['GENRE'],
            sourceNodeFilter: sourceNodeId,
            targetNodeFilter:'Movie'
        })
        YIELD node1, node2, similarity
        RETURN gds.util.asNode(node1).name AS Person1, gds.util.asNode(node2).name AS Person2, similarity
        ORDER BY similarity DESCENDING, Person1, Person2
        """     
    #Find Movies based on similar actors    
    elif sim_type.lower()=='actor':
        node = """
        MATCH (m:Movie)
        WHERE tolower(m.name) = '"""+entity.lower()+"""'
        WITH id(m) AS sourceNodeId
        CALL gds.nodeSimilarity.filtered.stream('movies2',{
            nodeLabels:['Movie','Person'],
            relationshipTypes:['ACTED_IN'],
            sourceNodeFilter: sourceNodeId,
            targetNodeFilter:'Movie'
        })
        YIELD node1, node2, similarity
        RETURN gds.util.asNode(node1).name AS Person1, gds.util.asNode(node2).name AS Person2, similarity
        ORDER BY similarity DESCENDING, Person1, Person2
        """  
    #Find Similar Personalities to a director    
    elif sim_type.lower()=='director':
        node = """
        MATCH (m:Movie)<-[c:CREWED_IN{character:'Directing'}]-(p:Person)
        WHERE tolower(m.name) = '""" + entity.lower() + """'
        WITH id(m) AS sourceNodeId,id(p) as directorNodeId
        CALL gds.nodeSimilarity.filtered.stream('movies3',{
            nodeLabels:['Movie','Person'],
            relationshipTypes:['CREWED_IN','ACTED_IN'],
            sourceNodeFilter: directorNodeId,
            targetNodeFilter: 'Person'
        })
        YIELD node1, node2, similarity
        RETURN gds.util.asNode(node1).name AS Person1, gds.util.asNode(node2).name AS Person2, similarity
        ORDER BY similarity DESCENDING, Person1, Person2
        """        
    #Find Similar Actors    
    elif sim_type.lower()=='similar actor':
        node = """
        MATCH (p:Person)
        WHERE tolower(p.name) = '""" + entity.lower() + """'
        WITH id(p) as actorNodeId
        CALL gds.nodeSimilarity.filtered.stream('movies3',{
            nodeLabels:['Movie','Person'],
            relationshipTypes:['ACTED_IN'],
            sourceNodeFilter: actorNodeId,
            targetNodeFilter: 'Person',
            topk:20
        })
        YIELD node1, node2, similarity
        RETURN gds.util.asNode(node1).name AS Person1, gds.util.asNode(node2).name AS Person2, similarity
        ORDER BY similarity DESCENDING, Person1, Person2
        """      
    #Find Non-Similar Actors      
    elif sim_type.lower()=='nonsimilar actor':
        node = """
        MATCH (p:Person)
        WHERE tolower(p.name) = '""" + entity.lower() + """'
        WITH id(p) as actorNodeId
        CALL gds.nodeSimilarity.filtered.stream('movies3',{
            nodeLabels:['Movie','Person'],
            relationshipTypes:['ACTED_IN'],
            sourceNodeFilter: actorNodeId,
            targetNodeFilter: 'Person',
            bottomk:20
        })
        YIELD node1, node2, similarity
        RETURN gds.util.asNode(node1).name AS Person1, gds.util.asNode(node2).name AS Person2, similarity
        ORDER BY similarity DESCENDING, Person1, Person2
        """
    #Find Movies based on Production Company   
    elif sim_type.lower()=='production_house':
        node = """
       MATCH (m:Movie)
        WHERE tolower(m.name) = '"""+entity.lower()+"""'
        WITH id(m) AS sourceNodeId
        CALL gds.nodeSimilarity.filtered.stream('movies2',{
            nodeLabels:['Movie','ProductionCompany'],
            relationshipTypes:['PRODUCED_BY'],
            sourceNodeFilter: sourceNodeId,
            targetNodeFilter: 'Movie'
        })
        YIELD node1, node2, similarity
        RETURN gds.util.asNode(node1).name AS Person1, gds.util.asNode(node2).name AS Person2, similarity
        ORDER BY similarity DESCENDING, Person1, Person2
        """  
    #Find Similar Movies by Region   
    elif sim_type.lower()=='country':
        node = """
       MATCH (m:Movie)
        WHERE tolower(m.name) = '"""+entity.lower()+"""'
        WITH id(m) AS sourceNodeId
        CALL gds.nodeSimilarity.filtered.stream('movies2',{
            nodeLabels:['Movie','Country'],
            relationshipTypes:['COUNTRY'],
            sourceNodeFilter: sourceNodeId,
            targetNodeFilter: 'Movie'
        })
        YIELD node1, node2, similarity
        RETURN gds.util.asNode(node1).name AS Person1, gds.util.asNode(node2).name AS Person2, similarity
        ORDER BY similarity DESCENDING, Person1, Person2
        """        
    #Find Similar Movies by Language   
    elif sim_type.lower()=='language':
        node = """
       MATCH (m:Movie)
        WHERE tolower(m.name) = '"""+entity.lower()+"""'
        WITH id(m) AS sourceNodeId
        CALL gds.nodeSimilarity.filtered.stream('movies2',{
            nodeLabels:['Movie','Country','SpokenLanguage'],
            relationshipTypes:['LANGUAGE'],
            sourceNodeFilter: sourceNodeId,
            targetNodeFilter: 'Movie'
        })
        YIELD node1, node2, similarity
        RETURN gds.util.asNode(node1).name AS Person1, gds.util.asNode(node2).name AS Person2, similarity
        ORDER BY similarity DESCENDING, Person1, Person2
        """                        
    #Find Similar Movies by Region and Language?   
    elif sim_type.lower()=='geography':
        node = """
       MATCH (m:Movie)
        WHERE tolower(m.name) = '"""+entity.lower()+"""'
        WITH id(m) AS sourceNodeId
        CALL gds.nodeSimilarity.filtered.stream('movies2',{
            nodeLabels:['Movie','Country','SpokenLanguage'],
            relationshipTypes:['LANGUAGE','COUNTRY'],
            sourceNodeFilter: sourceNodeId,
            targetNodeFilter: 'Movie'
        })
        YIELD node1, node2, similarity
        RETURN gds.util.asNode(node1).name AS Person1, gds.util.asNode(node2).name AS Person2, similarity
        ORDER BY similarity DESCENDING, Person1, Person2        
        """
    #Find Actors who have the worked the most with a director?   
    elif sim_type.lower()=='work':
        node = """
        MATCH (p:Person)
        WHERE tolower(p.name) = '""" + entity.lower() + """'
        WITH id(p) as actorNodeId
        CALL gds.nodeSimilarity.filtered.stream('movies3',{
            nodeLabels:['Movie','Person'],
            relationshipTypes:['CREWED_IN'],
            sourceNodeFilter: actorNodeId,
            targetNodeFilter: 'Person',
            topk:20
        })
        YIELD node1, node2, similarity
        RETURN gds.util.asNode(node1).name AS Person1, gds.util.asNode(node2).name AS Person2, similarity
        ORDER BY similarity DESCENDING, Person1, Person2
        """
    #General if sim_type is None       
    else:
        node = """
        MATCH (m:Movie)
        WHERE tolower(m.name) = '"""+entity.lower()+"""'
        WITH id(m) AS sourceNodeId
        CALL gds.nodeSimilarity.filtered.stream('movies2',{
            nodeLabels:['Movie','Person','User'],
            relationshipTypes:['CREWED_IN','ACTED_IN','RATING'],
            sourceNodeFilter: sourceNodeId,
            targetNodeFilter:'Movie'      
        })
        YIELD node1, node2, similarity
        RETURN gds.util.asNode(node1).name AS Person1, gds.util.asNode(node2).name AS Person2, similarity
        ORDER BY similarity DESCENDING, Person1, Person2
        """        
    print(node)
    return gds.run_cypher(node)    
'''

#Similar movies based on user rating
#Similar movies based on Genre
#Find Movies based on similar actors    
#Find Similar Personalities to a director 
#Find Similar Actors      
#Find Non-Similar Actors  
#Find Movies based on Production Company   
#Find Similar Movies by Region   
#Find Similar Movies by Language   
#Find a director an actor should work with ? 
#General if sim_type is None 
'''

df1 = getSimilar("harrison ford",'work')
#%%