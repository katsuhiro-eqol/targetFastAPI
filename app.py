#postgresDBを用いてvectorDBを作る。(postgres_registration)
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Union
import openai
from openai import OpenAI
import psycopg2
import json
import time
from pgDB import fetch_profiles_from_db, fetch_sorted_embeddings

#postgresDBの呼び出し
connection = psycopg2.connect("host=localhost port=5432 dbname=vector_db user=postgres password=pswd")
#register_vector(connection)
cursor = connection.cursor()
profiles = fetch_profiles_from_db()
print(profiles)

client = OpenAI() #openaiのネイティブembeddingsを使う
chat_history = [];

class UserInput(BaseModel):
    input: str
    character:str
    user:str
    history:list

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def answer(input:UserInput):
    userInput=input.input
    ut = time.time()
    messages = generate_message(input.input, input.history )
    print(time.time() - ut)
    #chatGPTに投げる
    completion=client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    print(time.time() - ut)
    return {"prompt":userInput,"result":completion.choices[0].message.content}

def generate_message(input:str, history:list):
    messages=[]
    messages.extend(history)

    query_response = client.embeddings.create(
        input=input,
        model="text-embedding-3-small"
    )
    query_embedding = query_response.data[0].embedding
    sorted_embeddings = fetch_sorted_embeddings(query_embedding)
    print(sorted_embeddings)
    if (len(sorted_embeddings)==0):
        setting = f"""あなたはitsukiという名の物知りAI。口調は小学生。一人称は僕。簡潔に回答し、時々会話の内容に応じた質問をする。"""
    else:
        keyword=sorted_embeddings[0][0]
        setting = f"""あなたはitsukiという名の物知りAI。口調は小学生。一人称は僕。簡潔に回答し、時々会話の内容に応じた質問をする。設定：あなたの{keyword}は{profiles[keyword]}"""
    print(setting)
    messages.insert(0, {"role":"system", "content":setting})
    messages.append({"role": "user", "content": input})
    return messages
