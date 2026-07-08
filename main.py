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
        "de responder, e nunca inventa informação que não está documentada."
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
    ),
    tools=[ferramenta_tecnica],
)


tarefa_atendimento = Task(
    description=(
        "Um funcionário perguntou: '{pergunta_funcionario}'. "
        "Determine se a pergunta é sobre política de RH ou sobre documentação "
        "técnica, delegue para o especialista correto, e retorne a resposta dele."
    ),
    expected_output="Uma resposta clara e precisa, baseada na documentação oficial correta.",
    agent=None,
)

crew = Crew(
    agents=[agente_rh, agente_tecnico],
    tasks=[tarefa_atendimento],
    process=Process.hierarchical,
    manager_llm="gpt-4o",
    verbose=True,
)

if __name__ == "__main__":
    resultado = crew.kickoff(
        inputs={"pergunta_funcionario": "Quantos dias posso trabalhar remoto?"}
    )
    print("\n=== RESULTADO FINAL ===")
    print(resultado)
