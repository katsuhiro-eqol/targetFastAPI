import psycopg2
from openai import OpenAI
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import time

ut = time.time()
client = OpenAI()

input="明日はどこに行きますか？"
response = client.embeddings.create(
    input=input,
    model="text-embedding-3-small"
)
query_embedding = response.data[0].embedding

#kw_embeddingsの呼び出し
connection = psycopg2.connect("host=localhost port=5432 dbname=vector_db user=postgres password=pswd")
cursor = connection.cursor()
cursor.execute("SELECT id, keyword, embedding FROM keyword_embeddings")
rows = cursor.fetchall()


columns = [desc[0] for desc in cursor.description]
data = pd.DataFrame(rows, columns=columns)

data['similarity'] = data['embedding'].apply(lambda emb: 1 - cosine(query_embedding, eval(emb)))
filtered_data = data[data['similarity']>0.28]

print(filtered_data[["keyword", "similarity"]])
print(time.time()-ut)

'''
matches = []
similarity_threshold = 0.1
num_matches = 3

query = """
	WITH vector_matches AS (
	  SELECT 
	    keyword, 
	    MAX(1 - (embedding <=> %s)) AS similarity
	  FROM kw_embeddings
	  WHERE 1 - (embedding <=> %s) > %s
	  ORDER BY similarity DESC
	  LIMIT %s
	)
    SELECT
      id,
      keyword,
      vector_matches.similarity
    ORDER BY
      vector_matches.similarity DESC;
    """

cursor.execute(query, (emb, emb, similarity_threshold, num_matches))
results = cur.fetchall()
columns = [desc[0] for desc in cursor.description]

for r in results:
    matches.append(
        {
            "id": r[columns.index("id")],
            "keyword": r[columns.index("keyword")],
            "simirality": r[columns.index("simirarity")]
        }
    )
print(matches)

#cursor.execute("SELECT * FROM kw_embeddings")
#print(cursor.rowcount)
'''
cursor.close()
connection.close()
