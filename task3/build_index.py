from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

def main():
    knowledge_base = Path(__file__).resolve().parent.parent / "knowledge_base"

    model = SentenceTransformer("intfloat/multilingual-e5-large")

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )
    try:
        client.delete_collection("knowledge_base")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name="knowledge_base"
    )

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=300,
        chunk_overlap=50
        )

    documents = []
    embedding_texts = []
    metadatas = []
    ids = []

    for file_path in knowledge_base.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        lines = text.splitlines()
        text = "\n".join(lines[1:]).strip()

        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            documents.append(chunk)

            embedding_texts.append(f"passage: {chunk}")

            metadatas.append({
                "source": file_path.name,
                "chunk": i
            })

            ids.append(f"{file_path.stem}_{i}")

    embeddings = model.encode(embedding_texts).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f"Добавлено документов: {len(documents)}")


    question = "Где находится крепость Редволл?"

    # Создаём embedding запроса
    query_embedding = model.encode(
        [f"query: {question}"]
    ).tolist()

    # Ищем 5 наиболее подходящих чанков
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5
    )

    # Выводим результаты
    for i, (document, metadata, distance) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ),
            1
    ):
        print(f"\n--- Результат {i} ---")
        print("Источник:", metadata["source"])
        print("Расстояние:", distance)
        print("Текст:", document)

if __name__ == "__main__":
    main()