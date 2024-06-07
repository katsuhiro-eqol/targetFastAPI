import psycopg2
from openai import OpenAI
from pgvector.psycopg2 import register_vector
import numpy as np
from scipy.spatial.distance import cosine
import time

# データベース接続設定
connection = psycopg2.connect("host=localhost port=5432 dbname=vector_db user=postgres password=pswd")
register_vector(connection)
cursor = connection.cursor()

# OpenAIクライアントの初期化
client = OpenAI()

# embeddingsを生成してPostgresDBに保存する関数
def save_embeddings_to_db():
    keywords = ["名前","年齢","誕生日","住所","家族","所属","趣味","好きなもの","苦手なもの"]
    for keyword in keywords:
        # OpenAIからembeddingを取得
        response = client.embeddings.create(
            input=keyword,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        
        # PostgresDBにembeddingを保存
        sql = "INSERT INTO keyword_embeddings (keyword, embedding) VALUES (%s, %s)"
        cursor.execute(sql, (keyword, embedding))
    
    # 変更をコミット
    connection.commit()
    print("Embeddingsがデータベースに保存されました。")

# プロファイル情報をデータベースに登録する関数
def save_profile_to_db():
    profiles = {
        "名前": "itsuki",
        "年齢": 100,
        "誕生日": "不明",
        "住所": "サイバー空間",
        "家族": "父はTransformer,母はLLM、兄はBERT",
        "所属": "OpenAI",
        "趣味": "カウンセリング、囲碁",
        "好きなもの": "チーズケーキ",
        "苦手なもの": "パクチー"
    }
    for key, value in profiles.items():
        sql = "UPDATE keyword_embeddings SET profile = %s WHERE keyword = %s"
        cursor.execute(sql, (value, key))
    connection.commit()
    print(cursor.rowcount)
    print("プロファイル情報がデータベースに保存されました。")

# プロファイル情報をデータベースから取得する関数
def fetch_profiles_from_db():
    cursor.execute("SELECT keyword, profile FROM keyword_embeddings")
    results = cursor.fetchall()
    return dict(results)

# 新たなembeddingとのコサイン類似度でソートしたリストを返す関数
def fetch_sorted_embeddings(query_embedding):
    cursor.execute("SELECT keyword, embedding FROM keyword_embeddings")
    results = cursor.fetchall()
    # コサイン類似度を計算
    filtered_results = []
    for result in results:
        similarity = 1 - cosine(query_embedding, result[1])
        if similarity >= 0.28:
            filtered_results.append((result[0], similarity))
    return filtered_results

# 関数の実行
#save_embeddings_to_db()
#save_profile_to_db()

# リソースのクリーンアップ
#cursor.close()
#connection.close()
