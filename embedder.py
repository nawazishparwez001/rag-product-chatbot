from sentence_transformers import SentenceTransformer

# We use this specific model because it's small (80MB), fast, and produces
# good quality embeddings for semantic search. It maps text to a 384-dimensional
# vector — meaning each chunk becomes a list of 384 numbers that represent its meaning.
MODEL_NAME = "all-MiniLM-L6-v2"

# Load the model once at module level — loading is slow (~1-2 seconds).
# By loading here, we avoid reloading it every time embed_chunks() is called.
# The first time this runs, it will download the model from HuggingFace (~80MB).
model = SentenceTransformer(MODEL_NAME)


def embed_chunks(chunks):
    """
    Take a list of chunk dicts and add an 'embedding' field to each one.

    An embedding is a list of 384 numbers. Two chunks that are semantically
    similar (talk about the same idea) will have embeddings that are close
    together in 384-dimensional space. This is what makes semantic search work —
    we find chunks whose embeddings are closest to the query's embedding.

    Input:  list of chunk dicts (from chunker.py), each with a 'text' field
    Output: same list, but each dict now also has an 'embedding' field
    """
    # Extract just the text from each chunk for batch processing.
    # We batch all texts together because the model processes them much faster
    # in one batch than one at a time.
    texts = [chunk["text"] for chunk in chunks]

    print(f"Embedding {len(texts)} chunks using '{MODEL_NAME}'...")
    print("(First run will download the model — this may take a minute)")

    # encode() returns a numpy array of shape (num_chunks, 384)
    # show_progress_bar=True prints a progress bar so you can see it working
    embeddings = model.encode(texts, show_progress_bar=True)

    # Attach each embedding back to its corresponding chunk dict.
    # We convert to a plain Python list because ChromaDB (Phase 4) expects lists,
    # not numpy arrays.
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    print(f"\nDone. Each chunk now has an embedding of {len(embeddings[0])} dimensions.")
    return chunks


# Quick test: run this file directly to verify embeddings are being generated
if __name__ == "__main__":
    from loader import load_documents
    from chunker import chunk_documents

    print("Loading documents...")
    docs = load_documents()

    print("\nChunking documents...")
    chunks = chunk_documents(docs)

    # Only embed the first 5 chunks for testing — no need to embed all 2726
    # just to verify it works
    print("\nEmbedding first 5 chunks as a test...")
    sample = embed_chunks(chunks[:5])

    print("\n--- Sample Embedding ---")
    print(f"Chunk text  : {sample[0]['text'][:100]}...")
    print(f"Embedding   : {sample[0]['embedding'][:5]}... (showing first 5 of 384 numbers)")
    print(f"Total dims  : {len(sample[0]['embedding'])}")
