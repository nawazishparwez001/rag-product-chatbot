import re

# Chunk size in characters (not tokens) — we use characters here because
# splitting by tokens requires a tokenizer library. Characters are a good
# enough approximation at this stage: ~500 tokens ≈ ~2000 characters.
CHUNK_SIZE = 2000

# Overlap in characters — we repeat the last 200 characters of each chunk
# at the start of the next one. This prevents a sentence that spans a
# boundary from being lost or cut in half.
CHUNK_OVERLAP = 200


def split_into_chunks(text):
    """
    Split a long text into overlapping chunks of roughly CHUNK_SIZE characters.

    Why overlap? Imagine a key insight spans the end of chunk 3 and the start
    of chunk 4. Without overlap, no single chunk contains the full thought,
    and retrieval might miss it entirely. Overlap ensures continuity.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        # If we're not at the end of the text, try to break at a sentence
        # boundary (period + space) rather than mid-sentence.
        # Why? A chunk ending at a natural sentence break is more coherent
        # when read by the LLM during generation.
        if end < len(text):
            # Look for the last sentence-ending punctuation before the cutoff
            boundary = text.rfind(". ", start, end)
            if boundary != -1:
                end = boundary + 1  # include the period

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Move start forward by (CHUNK_SIZE - CHUNK_OVERLAP) so the next
        # chunk begins inside the current one — creating the overlap window.
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def chunk_documents(documents):
    """
    Take the list of documents from loader.py and return a new list of chunks.

    Each chunk is a smaller piece of a document, but it carries the parent
    document's metadata (title, type, date, source_url) so we never lose
    track of where a chunk came from. This is important for citing sources
    in the final answer.

    Input:  list of dicts with 'content', 'title', 'type', 'date', 'source_url'
    Output: list of dicts with 'text', 'title', 'type', 'date', 'source_url', 'chunk_index'
    """
    all_chunks = []

    for doc in documents:
        text_chunks = split_into_chunks(doc["content"])

        for i, chunk_text in enumerate(text_chunks):
            all_chunks.append({
                # The actual text that will be embedded and searched
                "text": chunk_text,

                # Metadata carried over from the parent document.
                # We keep this so when a chunk is retrieved, we can tell
                # the user "this came from the podcast with Albert Cheng, 2025-10-05"
                "title": doc["title"],
                "type": doc["type"],
                "date": doc["date"],
                "source_url": doc["source_url"],

                # Which chunk number within this document (useful for debugging)
                "chunk_index": i,
            })

    return all_chunks


# Quick test: run this file directly to see chunking in action
if __name__ == "__main__":
    # We import loader here only for testing — in the full pipeline,
    # loader and chunker will be called in sequence from a main script
    from loader import load_documents

    print("Loading documents...")
    docs = load_documents()

    print("\nChunking documents...")
    chunks = chunk_documents(docs)

    print(f"\nTotal chunks created: {len(chunks)}")
    print(f"From {len(docs)} documents")
    print(f"Average chunks per document: {len(chunks) / len(docs):.1f}")

    # Preview the first chunk
    print("\n--- Sample Chunk ---")
    c = chunks[0]
    print(f"Title      : {c['title']}")
    print(f"Type       : {c['type']}")
    print(f"Chunk index: {c['chunk_index']}")
    print(f"Length     : {len(c['text'])} characters")
    print(f"Text preview:\n{c['text'][:400]}...")
