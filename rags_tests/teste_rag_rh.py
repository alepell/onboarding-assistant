from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("docs/manual_rh.txt", "r", encoding="utf-8") as f:
    texto_rh = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks_rh = splitter.split_text(texto_rh)

print(f"Total de chunks RH: {len(chunks_rh)}")
for chunk in chunks_rh:
    print("---")
    print(chunk)


from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

modelo_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store_rh = Chroma.from_texts(chunks_rh, modelo_embeddings)

print("\nVector store RH criado com sucesso!")

retriever_rh = vector_store_rh.as_retriever(search_kwargs={"k": 2})

pergunta_teste = "Quantos dias posso trabalhar remoto?"

chunks_relevantes = retriever_rh.invoke(pergunta_teste)

print(f"\n=== Chunks relevantes para: '{pergunta_teste}' ===")
for chunk in chunks_relevantes:
    print("---")
    print(chunk.page_content)
