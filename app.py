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
    input_variables: str

app = FastAPI()

@app.post("/")
async def answer(input:UserInput):
    return input

@app.get("/")
async def root(input: Union[str, None] = None):
    if input:       
        output = taskExplain(input)
        #print(taskYomi(output))
        return {"解説":output}
    else:
        return {"解説":"入力されていません"}

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