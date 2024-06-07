from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Union

class UserInput(BaseModel):
    input: str

app = FastAPI()

@app.post("/")
async def answer(input:UserInput):
    return {"result":input}

@app.get("/")
async def root(input: Union[str, None] = None):
    if input:       
        return {"result":input}
    else:
        return {"result":"入力されていません"}

'''
langchain
langchain-community
langchain-core
openai
'''

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Union
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import SimpleSequentialChain
from langchain.memory import ConversationBufferMemory

chat1 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
chat2 = ChatOpenAI(model_name="ft:gpt-3.5-turbo-0613:personal::7yhcFCbA", temperature=0.4) #silva

class UserInput(BaseModel):
    input: str

app = FastAPI()

@app.post("/")
async def answer(input:UserInput):
    return input

@app.get("/")
async def root(input: Union[str, None] = None):
    if input:       
        output = taskExplain(input)
        #print(taskYomi(output))
        return {"result":output}
    else:
        return {"result":"入力されていません"}

@app.get("/memory")
async def memory(input: Union[str, None] = None):
    print(input)
    if input:
        output = taskMemory(input)
        return output
    else:
        return "inputが入力されていません"


def taskExplain(input):
    prompt1 = PromptTemplate(
        input_variables=["input"],
        template="次の語を50字以内で解説して。{input}",
        #template="あなたは克宏という名のAIコンサルタント。{input}",
    )
    chain1 = LLMChain(llm=chat1, prompt=prompt1)
    prompt2 = PromptTemplate(
        input_variables=["output"],
        template="次の読みをひらがなで表せ。{output}",
    )
    chain2 = LLMChain(llm=chat1, prompt=prompt2)
    overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)

    return overall_chain(input)

def taskMemory(input):
    template = """あなたは人間と話すチャットボットです。
    {chat_history}
    Human: {input}
    Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "input"],
        template=template
    )
    memory = ConversationBufferMemory(memory_key="chat_history")
    print(memory)
    llm_chain = LLMChain(
        llm=chat1,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )
    output = llm_chain.invoke(input)["text"]
    print(llm_chain.invoke(input))

    return output

    @app.post("/")
async def answer(input:UserInput):
    input=input.input
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "あなたはAIアシスタントです。５０字以内で簡潔に回答してください。"},
        {"role": "user", "content": input}
    ]
    )
    output=completion.choices[0].message
    print(output.content)
    return {"result": output.content}

    #chromadb
    import chromadb
import openai
import numpy as np
from chromadb.utils import embedding_functions
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
#import openai_schema
import os
import time

#chromaはsqlite。


if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print ("OPENAI_API_KEY is ready")
else:
    print ("OPENAI_API_KEY environment variable not found")


ut = time.time()

chroma_client = chromadb.PersistentClient(path="./chromaDB")
collection = chroma_client.get_or_create_collection("vector_collection")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
input="明日はどこに行きますか？"
query=collection.query(
    query_embeddings=openai_ef([input]),
    n_results=1
)
print(query["documents"][0][0])
print(time.time()-ut)
'''
keywords = ["名前","年齢","誕生日","住所","家族","所属","趣味","好きなもの","苦手なもの"]
m=collection.count()
print(m)
print(collection.peek(1)["documents"])

collection.add(
    embeddings=openai_ef(keywords),
    documents=keywords,
    ids=[str(n) for n in list(range(m,m+len(keywords)))]
)

output=collection.query(
    query_embeddings=openai_ef(["散歩は好きですか？"]),
    n_results=2
)
print(m)
print(collection.peek(1)["documents"])
#print(np.array(openai_ef(["This is a document","test"])).shape)
'''