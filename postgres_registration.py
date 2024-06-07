#vector_db kw_embeddingsにkeywordのvector情報を登録するスクリプト
#以下を参考にpostgresDDをDockerで立ち上げ。ただしpsycopg2を用いてできるだけシンプルに作る。
#https://qiita.com/kanaza-s/items/b46214ba8543e34c5003

import psycopg2
from openai import OpenAI
from pgvector.psycopg2 import register_vector

connection = psycopg2.connect("host=localhost port=5432 dbname=vector_db user=postgres password=pswd")
register_vector(connection)
cursor = connection.cursor()

client = OpenAI()

def keyword_registration():
    keywords = ["名前","年齢","誕生日","住所","家族","所属","趣味","好きなもの","苦手なもの"]
    for keyword in keywords:
        response = client.embeddings.create(
            input=keyword,
            model="text-embedding-3-small"
        )
        emb = response.data[0].embedding
        sql = "INSERT INTO keyword_embeddings (keyword, embedding) VALUES (%s, %s)"
        cursor.execute(sql, (keyword, emb))
        connection.commit()

    cursor.execute("SELECT * FROM keyword_embeddings")
    print(cursor.rowcount)
    cursor.close()
    connection.close()

def profile_registration():
    profiles = {"名前":"itsuki","年齢":100,"誕生日":"不明","住所":"サイバー空間","家族":"父はTransformer,母はLLM、兄はBERT","所属":"OpenAI","趣味":"カウンセリング、囲碁","好きなもの":"チーズケーキ","苦手なもの":"パクチー"}

def count_records():
    cursor.execute("SELECT * FROM keyword_embeddings")
    print(cursor.rowcount)
    cursor.close()
    connection.close()

#count_records()   
keyword_registration()


