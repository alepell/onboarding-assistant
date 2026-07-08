from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("docs/docs_tecnicos.txt", "r", encoding="utf-8") as f:
    texto_rh = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks_tecnicos = splitter.split_text(texto_rh)

print(f"Total de chunks Tecnicos: {len(chunks_tecnicos)}")
for chunk in chunks_tecnicos:
    print("---")
    print(chunk)


from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

modelo_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store_tecnicos = Chroma.from_texts(chunks_tecnicos, modelo_embeddings)

print("\nVector store Tecnicos criado com sucesso!")

retriever_tecnicos = vector_store_tecnicos.as_retriever(search_kwargs={"k": 2})

pergunta_teste = "Como eu solicito acesso a VPN?"

chunks_relevantes = retriever_tecnicos.invoke(pergunta_teste)

print(f"\n=== Chunks relevantes para: '{pergunta_teste}' ===")
for chunk in chunks_relevantes:
    print("---")
    print(chunk.page_content)
