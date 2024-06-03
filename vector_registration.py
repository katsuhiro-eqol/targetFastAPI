import chromadb
import openai
import numpy as np
from chromadb.utils import embedding_functions
from langchain.embeddings.openai import OpenAIEmbeddings
#import openai_schema
import os

if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print ("OPENAI_API_KEY is ready")
else:
    print ("OPENAI_API_KEY environment variable not found")

chroma_client = chromadb.PersistentClient(path="./chromaDB")
collection = chroma_client.get_or_create_collection("vector_collection")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
keywords = ["名前","年齢","誕生日","住所","家族","所属","趣味","好きなもの","苦手なもの"]
m=collection.count()
print(m)
print(collection.peek(1)["documents"])
'''
collection.add(
    embeddings=openai_ef(keywords),
    documents=keywords,
    ids=[str(n) for n in list(range(m,m+len(keywords)))]
)
'''
output=collection.query(
    query_embeddings=openai_ef(["散歩は好きですか？"]),
    n_results=2
)
print(m)
print(collection.peek(1)["documents"])
#print(np.array(openai_ef(["This is a document","test"])).shape)
