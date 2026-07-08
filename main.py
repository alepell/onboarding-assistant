from dotenv import load_dotenv
from rags.rag_rh import rag_rh
from rags.rag_tech import rag_tech


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

modelo_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# --- RAG do RH ---
retriever_rh = rag_rh("docs/manual_rh.txt", modelo_embeddings, splitter, Chroma)

# --- RAG Técnico ---
retriever_tecnico = rag_tech(
    "docs/docs_tecnicos.txt", modelo_embeddings, splitter, Chroma
)

print("Ambos os vector stores criados com sucesso!")


from crewai.tools import BaseTool


class FerramentaConsultaRH(BaseTool):
    name: str = "Consulta Política de RH"
    description: str = (
        "Use esta ferramenta para responder perguntas sobre política de RH: "
        "horário de trabalho, home office, benefícios, férias e avaliação de desempenho."
    )

    def _run(self, pergunta: str) -> str:
        chunks_relevantes = retriever_rh.invoke(pergunta)
        contexto = "\n\n".join([chunk.page_content for chunk in chunks_relevantes])
        return contexto


class FerramentaConsultaTecnica(BaseTool):
    name: str = "Consulta Documentação Técnica"
    description: str = (
        "Use esta ferramenta para responder perguntas sobre ferramentas internas: "
        "VPN, Git, acesso a bancos de dados, CI/CD e política de senhas."
    )

    def _run(self, pergunta: str) -> str:
        chunks_relevantes = retriever_tecnico.invoke(pergunta)
        contexto = "\n\n".join([chunk.page_content for chunk in chunks_relevantes])
        return contexto


ferramenta_rh = FerramentaConsultaRH()
ferramenta_tecnica = FerramentaConsultaTecnica()

from crewai import Agent, Task, Crew, Process

agente_rh = Agent(
    role="Especialista em RH",
    goal="Responder dúvidas de funcionários sobre políticas de RH com precisão",
    backstory=(
        "Você é um especialista de RH que conhece profundamente as políticas "
        "internas da empresa. Você sempre consulta a documentação oficial antes "
        "de responder, e nunca inventa informação que não está documentada. "
        "Se a pergunta não for sobre política de RH, você recusa educadamente "
        "e explica que não é da sua área."
    ),
    tools=[ferramenta_rh],
)

agente_tecnico = Agent(
    role="Especialista Técnico",
    goal="Responder dúvidas de funcionários sobre ferramentas e processos técnicos internos",
    backstory=(
        "Você é um especialista técnico que conhece profundamente a infraestrutura "
        "e ferramentas internas da empresa. Você sempre consulta a documentação "
        "oficial antes de responder, e nunca inventa informação que não está documentada."
        "Se a pergunta não for sobre política de tecnologia, você recusa educadamente "
        "e explica que não é da sua área."
    ),
    tools=[ferramenta_tecnica],
)

tarefa_atendimento = Task(
    description=(
        "Um funcionário perguntou: '{pergunta_funcionario}'. "
        "Determine se a pergunta é sobre política de RH ou sobre documentação "
        "técnica interna da empresa, delegue para o especialista correto, e "
        "retorne a resposta dele. "
        "IMPORTANTE: se a pergunta não tiver relação com RH ou com a "
        "documentação técnica interna da empresa, NÃO tente responder usando "
        "conhecimento geral. Em vez disso, responda apenas: "
        "'Essa pergunta está fora do escopo deste assistente, que cobre apenas "
        "política de RH e documentação técnica interna da empresa.'"
    ),
    expected_output=(
        "Uma resposta clara baseada na documentação oficial correta, OU a "
        "mensagem padrão de fora de escopo, se a pergunta não se encaixar."
    ),
    agent=None,
)


crew = Crew(
    agents=[agente_rh, agente_tecnico],
    tasks=[tarefa_atendimento],
    process=Process.hierarchical,
    manager_llm="gpt-4o",
    verbose=True,
)


def responder_pergunta(pergunta: str) -> str:
    resultado = crew.kickoff(inputs={"pergunta_funcionario": pergunta})
    return str(resultado)


if __name__ == "__main__":
    resposta = responder_pergunta("Como eu solicito acesso à VPN?")
    print("\n=== RESULTADO FINAL ===")
    print(resposta)
