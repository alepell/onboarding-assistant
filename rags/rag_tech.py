def rag_tech(file_path: str, embeddings_model, splitter, chroma_class):
    with open(file_path, "r", encoding="utf-8") as f:
        texto_tecnico = f.read()

    chunks_tecnico = splitter.split_text(texto_tecnico)
    vector_store_tecnico = chroma_class.from_texts(chunks_tecnico, embeddings_model)
    retriever_tecnico = vector_store_tecnico.as_retriever(search_kwargs={"k": 2})
    return retriever_tecnico
