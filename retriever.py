from embedder import model
from store import get_collection

# How many chunks to retrieve for each query.
# 5 is a good default — enough context for Claude to work with,
# but not so many that we overwhelm the LLM's context window.
TOP_K = 5


def retrieve(query):
    """
    Given a user's question, find the most semantically relevant chunks
    from the vector database.

    Why semantic search instead of keyword search?
    Keyword search finds exact word matches. Semantic search finds meaning matches.
    For example, "how to grow a product" and "strategies for user growth" have
    no words in common but are semantically close — semantic search finds that.

    Returns a list of dicts, each with 'text', 'title', 'type', 'date', 'source_url'.
    """
    collection = get_collection()

    # Embed the query using the same model we used for the chunks.
    # This is critical — the query and chunks must live in the same vector space
    # for the similarity comparison to be meaningful.
    query_embedding = model.encode(query).tolist()

    # Ask ChromaDB to find the TOP_K chunks whose embeddings are closest
    # to the query embedding. ChromaDB uses cosine similarity by default.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    # Unpack ChromaDB's response format into clean, readable dicts.
    # ChromaDB returns nested lists (one per query), so we take index [0]
    # since we only sent one query at a time.
    chunks = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": text,
            "title": metadata["title"],
            "type": metadata["type"],
            "date": metadata["date"],
            "source_url": metadata["source_url"],
            # Distance is how "far apart" the vectors are — lower = more similar.
            # We convert to a similarity score (0 to 1) so it's easier to read.
            "similarity": round(1 - distance, 3),
        })

    return chunks


# Test retrieval with a sample question
if __name__ == "__main__":
    query = "How should a PM think about product growth?"

    print(f"Query: {query}\n")
    print("Top matching chunks:\n")

    results = retrieve(query)

    for i, chunk in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(f"Title      : {chunk['title']}")
        print(f"Type       : {chunk['type']}")
        print(f"Similarity : {chunk['similarity']}")
        print(f"Preview    : {chunk['text'][:200]}...")
        print()
