import os
from dotenv import load_dotenv
import json
import base64
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
import functools
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from tqdm import tqdm
import math
import vertexai
from vertexai.language_models import TextEmbeddingModel
import numpy as np
import matplotlib.pyplot as plt
# import mplcursors

def authenticate():
    # return "credentials"
    #Load .env
    load_dotenv()
    
    #Decode key and store in .JSON
    key_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    # print(key_path)
    # Create credentials based on key from service account
    # Make sure your account has the roles listed in the Google Cloud Setup section
    credentials = Credentials.from_service_account_file(
        key_path,
        scopes=['https://www.googleapis.com/auth/cloud-platform'])

    if credentials.expired:
        credentials.refresh(Request())
    
    #Set project ID accoridng to environment variable    
    # PROJECT_ID = os.getenv('PROJECT_ID')
        
    return credentials



def generate_batches(sentences, batch_size = 5):
    for i in range(0, len(sentences), batch_size):
        yield sentences[i : i + batch_size]

def encode_texts_to_embeddings(sentences):
    model = TextEmbeddingModel.from_pretrained(
        "textembedding-gecko@001")
    try:
        embeddings = model.get_embeddings(sentences)
        return [embedding.values for embedding in embeddings]
    except Exception:
        return [None for _ in range(len(sentences))]
        
def encode_text_to_embedding_batched(sentences, api_calls_per_second = 10, batch_size = 5):
    # Generates batches and calls embedding API
    
    embeddings_list = []

    # Prepare the batches using a generator
    batches = generate_batches(sentences, batch_size)

    seconds_per_job = 1 / api_calls_per_second
    print('number of batches:', math.ceil(len(sentences) / batch_size))
    print('estimated time:', math.ceil(len(sentences) / batch_size) * seconds_per_job)
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for batch in tqdm(
            batches, total = math.ceil(len(sentences) / batch_size), position=0
        ):
            futures.append(
                executor.submit(functools.partial(encode_texts_to_embeddings), batch)
            )
            # print('progress:', len(futures) / math.ceil(len(sentences) / batch_size) * 100, '%')
            time.sleep(seconds_per_job)
            
            if len(futures) >= 50:  # Adjust this value as needed
                for future in as_completed(futures):
                    embeddings_list.extend(future.result())
                futures = []

        for future in tqdm(futures):
            # print('-' * 10, 'future done', '-' * 10)
            embeddings_list.extend(future.result())
            
    print('all batches done')
    # is_successful = [
    #     embedding is not None for sentence, embedding in zip(sentences, embeddings_list)
    # ]
    embeddings_list_successful = np.squeeze(
        np.stack([embedding for embedding in tqdm(embeddings_list) if embedding is not None])
    )
    return embeddings_list_successful
