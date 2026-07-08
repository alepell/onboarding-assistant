from fastapi import FastAPI
from pydantic import BaseModel

from main import responder_pergunta

app = FastAPI()


class PerguntaRequest(BaseModel):
    pergunta: str


@app.post("/perguntar")
def perguntar(request: PerguntaRequest):
    resposta = responder_pergunta(request.pergunta)
    return {"resposta": resposta}
