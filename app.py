from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Union
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import chromadb
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from chromadb.utils import embedding_functions
import openai
import json
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=3)
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.4)
#chat2 = ChatOpenAI(model_name="ft:gpt-3.5-turbo-0613:personal::7yhcFCbA", temperature=0.4) #silva

#vector用のDB
chroma_client = chromadb.PersistentClient(path="./chromaDB")
collection = chroma_client.get_or_create_collection("vector_collection")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-3-small")

keywords = ["名前","年齢","誕生日","住所","家族","所属","趣味","好きなもの","苦手なもの"]
profiles = {"名前":"itsuki","年齢":100,"誕生日":"不明","住所":"サイバー空間","家族":"父はTransformer,母はLLM、兄はBERT","所属":"OpenAI","趣味":"カウンセリング、囲碁","好きなもの":"チーズケーキ","苦手なもの":"パクチー"}

class UserInput(BaseModel):
    input: str
    character:str
    user:str

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
    query=collection.query(
        query_embeddings=openai_ef([userInput]),
        n_results=1
    )
    keyword=query["documents"][0][0]
    setting = f"""あなたはitsukiという名のAI。設定に基づいて簡潔に回答すること。設定：あなたの{keyword}は{profiles[keyword]}"""
    print(setting)

    history=memory.load_memory_variables({})
    messages=[]
    messages.extend(memory.load_memory_variables({})["chat_history"])
    messages.insert(0,SystemMessage(content=setting))
    messages.append(HumanMessage(content=userInput))

    #chatGPTに投げる
    #output=chat.invoke(messages)
    #output=chat.invoke(messages, functions=[UserDetails.openai_schema], function_call={"name": UserDetails.openai_schema["name"]})
    output2=chat.invoke(messages)
    memory.chat_memory.add_user_message(userInput)
    memory.chat_memory.add_ai_message(output2.content)

    print(memory.load_memory_variables({})["chat_history"])
    #"arguments = json.loads(output.additional_kwargs["function_call"]["arguments"])
    #print(arguments)

    return {"prompt":userInput,"result":output2.content}

#spaCyでユーザーデータを取得するのは精度が悪い。一まとまりの会話からChatGPTにやらせる方が良い。例えば次のプロンプトが使える。
#以下の会話から、「Aの[name, age, address, birthday, hobby, favorite foods]を抽出して、JSONで回答せよ」のようなこと。
#firebaseのfunctionsで一日の終わりに処理する
'''
@app.post("/getUserData")
async def getUserData(input:UserInput):
    userInput=input.input
    doc = nlp(userInput)
    print([(ent.text, ent.label_) for ent in doc.ents]) 
    return {"arguments":"test3"}
'''