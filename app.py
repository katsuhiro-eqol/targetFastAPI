from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Union

class UserInput(BaseModel):
    input: str

app = FastAPI()

@app.post("/")
async def answer(input:UserInput):
    result = input.input
    return {"result": result}

@app.get("/")
async def root(input: Union[str, None] = None):
    if input:       

        return {"result":input}
    else:
        return {"result":"入力されていません"}
