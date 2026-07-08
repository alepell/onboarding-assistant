# 🧭 Onboarding Assistant

Assistente de onboarding para novos funcionários, construído com **agentes de IA especializados** que respondem dúvidas sobre **política de RH** e **documentação técnica interna**, sempre com base nos documentos oficiais da empresa — nunca "inventando" respostas.

## ✨ Como funciona

O projeto usa [CrewAI](https://www.crewai.com/) para orquestrar dois agentes especialistas sob um processo hierárquico:

```
┌─────────────────────────┐
│   Pergunta do usuário    │
└────────────┬─────────────┘
             │
             ▼
     ┌───────────────┐        gpt-4o decide para quem delegar
     │  Agente Gestor │        (ou recusa se estiver fora de escopo)
     └───────┬───────┘
             │
     ┌───────┴────────┐
     ▼                ▼
┌──────────┐    ┌──────────────┐
│ Agente RH │    │ Agente Técnico│
└─────┬─────┘    └───────┬──────┘
      │                  │
      ▼                  ▼
 RAG (manual_rh.txt) RAG (docs_tecnicos.txt)
```

1. Cada agente possui uma **ferramenta de RAG** própria, que busca os trechos mais relevantes em uma base vetorial ([Chroma](https://www.trychroma.com/)) criada a partir dos documentos internos.
2. O **Agente de RH** responde sobre horário de trabalho, home office, benefícios, férias e avaliação de desempenho.
3. O **Agente Técnico** responde sobre VPN, Git, acesso a bancos de dados, CI/CD e política de senhas.
4. Se a pergunta não se encaixar em nenhum dos dois temas, o assistente recusa educadamente em vez de responder com conhecimento genérico.

## 📂 Estrutura do projeto

```
onboarding-assistant/
├── main.py                  # Define os agentes, ferramentas e a crew
├── api.py                   # API REST (FastAPI) para expor o assistente
├── docs/
│   ├── manual_rh.txt        # Base de conhecimento de RH
│   └── docs_tecnicos.txt    # Base de conhecimento técnica
├── rags/
│   ├── rag_rh.py             # Monta o vector store / retriever de RH
│   └── rag_tech.py           # Monta o vector store / retriever técnico
└── rags_tests/               # Scripts para testar os RAGs isoladamente
```

## 🚀 Pré-requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) para gerenciamento de dependências
- Uma chave de API da OpenAI

## ⚙️ Instalação

```bash
# Clone o repositório
git clone <url-do-repositorio>
cd onboarding-assistant

# Instale as dependências
uv sync
```

Crie o arquivo `.env` a partir do exemplo e preencha suas chaves:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sua-chave-aqui
```

## ▶️ Uso

### Linha de comando

Executa uma pergunta de exemplo diretamente:

```bash
uv run main.py
```

### API REST

Suba o servidor FastAPI:

```bash
uv run uvicorn api:app --reload
```

Faça uma pergunta via `POST /perguntar`:

```bash
curl -X POST http://localhost:8000/perguntar \
  -H "Content-Type: application/json" \
  -d '{"pergunta": "Como eu solicito acesso à VPN?"}'
```

Resposta:

```json
{
  "resposta": "Para solicitar acesso à VPN, você deve abrir uma solicitação no portal ServiceNow, na categoria \"Acesso Remoto\". A aprovação leva até 24h úteis."
}
```

A documentação interativa da API fica disponível em `http://localhost:8000/docs`.

## 🧪 Testando os RAGs isoladamente

```bash
uv run rags_tests/teste_rag_rh.py
uv run rags_tests/teste_rag_tecnico.py
```

Esses scripts geram os chunks dos documentos, criam o vector store e mostram os trechos mais relevantes recuperados para uma pergunta de teste.

## 🛠️ Stack técnica

| Camada              | Tecnologia                                      |
| ------------------- | ------------------------------------------------ |
| Orquestração de agentes | [CrewAI](https://www.crewai.com/)             |
| LLM                 | OpenAI `gpt-4o`                                   |
| Embeddings          | OpenAI `text-embedding-3-small`                   |
| Vector store         | [Chroma](https://www.trychroma.com/)             |
| Chunking            | LangChain `RecursiveCharacterTextSplitter`        |
| API                 | [FastAPI](https://fastapi.tiangolo.com/)          |
