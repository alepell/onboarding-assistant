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
