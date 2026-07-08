def rag_rh(file_path: str, embeddings_model, splitter, chroma_class):
    with open(file_path, "r", encoding="utf-8") as f:
        texto_rh = f.read()

    chunks_rh = splitter.split_text(texto_rh)
    vector_store_rh = chroma_class.from_texts(chunks_rh, embeddings_model)
    retriever_rh = vector_store_rh.as_retriever(search_kwargs={"k": 2})
    return retriever_rh
