import chromadb

# Where ChromaDB will save its files on disk.
# Using a relative path means it creates a chroma_db/ folder inside
# wherever you run this script from — i.e. your project folder.
CHROMA_PATH = "./chroma_db"

# A "collection" in ChromaDB is like a table in a regular database.
# All our chunks live in one collection.
COLLECTION_NAME = "lenny_content"


def get_collection():
    """
    Connect to (or create) the ChromaDB collection on disk.

    PersistentClient means data is saved to disk and survives restarts.
    If the chroma_db/ folder doesn't exist yet, ChromaDB creates it.
    If it already exists, ChromaDB loads the existing data.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # get_or_create_collection means:
    # - First run: creates a new empty collection
    # - Subsequent runs: loads the existing collection (no re-embedding needed)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def store_chunks(chunks):
    """
    Save all embedded chunks into ChromaDB.

    ChromaDB expects four parallel lists:
    - ids        : unique string ID for each chunk
    - embeddings : the 384-number vectors
    - documents  : the raw text (ChromaDB calls text "documents")
    - metadatas  : dicts of extra info (title, date, type, source_url)

    We only store chunks if the collection is empty — this prevents
    re-inserting duplicates every time you run the script.
    """
    collection = get_collection()

    # If chunks are already stored, skip — no need to re-embed and re-store
    if collection.count() > 0:
        print(f"Collection already has {collection.count()} chunks. Skipping storage.")
        return collection

    print(f"Storing {len(chunks)} chunks into ChromaDB...")

    # Build the four parallel lists ChromaDB expects
    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        # ID must be a unique string — we combine source file + chunk index
        # so it's meaningful and debuggable, not just a random number
        chunk_id = f"{chunk['type']}_{i}_{chunk['chunk_index']}"

        ids.append(chunk_id)
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
        metadatas.append({
            "title": chunk["title"],
            "type": chunk["type"],
            "date": chunk["date"],
            "source_url": chunk["source_url"],
            "chunk_index": chunk["chunk_index"],
        })

    # ChromaDB has a limit on how many items you can add in one call.
    # We batch in groups of 500 to stay safe and show progress.
    batch_size = 500
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  Stored chunks {start} to {min(end, len(ids))}...")

    print(f"\nDone. Total chunks in DB: {collection.count()}")
    return collection


# Run this file directly to build the full vector database.
# This is a one-time operation — after this, chroma_db/ persists on disk.
if __name__ == "__main__":
    from loader import load_documents
    from chunker import chunk_documents
    from embedder import embed_chunks

    print("Step 1/3: Loading documents from GitHub...")
    docs = load_documents()

    print("\nStep 2/3: Chunking documents...")
    chunks = chunk_documents(docs)

    print("\nStep 3/3: Embedding and storing chunks...")
    chunks_with_embeddings = embed_chunks(chunks)
    store_chunks(chunks_with_embeddings)

    print("\nVector database is ready at ./chroma_db/")
