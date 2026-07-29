import chromadb
from sentence_transformers import SentenceTransformer

# Loads a small, fast, free local embedding model (runs on your machine, no API call needed)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Persistent client saves the vector database to disk, so you don't need to
# re-embed the PDF every single time you run the script
chroma_client = chromadb.PersistentClient(path="./chroma_db")


def get_or_create_collection(name="pdf_chunks"):
    """A 'collection' in ChromaDB is like a table, a named group of stored vectors."""
    return chroma_client.get_or_create_collection(name=name)


def embed_chunks(chunks):
    """Converts a list of text chunks into a list of vector embeddings."""
    embeddings = embedding_model.encode(chunks).tolist()
    return embeddings


def store_chunks(chunks, collection_name="pdf_chunks", source_name="document"):
    """Embeds and stores chunks in ChromaDB with unique IDs."""
    collection = get_or_create_collection(collection_name)
    embeddings = embed_chunks(chunks)

    ids = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,  # the original text, so we can retrieve it later
        embeddings=embeddings,  # the vector representation
        ids=ids,  # unique identifier per chunk
    )
    print(f"Stored {len(chunks)} chunks in collection '{collection_name}'.")
    return collection


def search_similar_chunks(query, collection_name="pdf_chunks", top_k=3):
    """
    Embeds the user's query, then finds the top_k most similar chunks
    using semantic (dense) similarity search.
    """
    collection = get_or_create_collection(collection_name)
    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    # results["documents"][0] is a list of the top_k matching chunk texts
    return results["documents"][0]


if __name__ == "__main__":
    from pdf_loader import load_and_chunk_pdf

    pdf_path = input("Enter path to a PDF file: ")
    chunks = load_and_chunk_pdf(pdf_path, chunk_size=150, overlap=30)

    store_chunks(chunks, source_name="handbook")

    query = input("\nEnter a test question: ")
    top_chunks = search_similar_chunks(query)

    print(f"\n--- Top {len(top_chunks)} matching chunks ---\n")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"[{i}] {chunk[:300]}...\n")
