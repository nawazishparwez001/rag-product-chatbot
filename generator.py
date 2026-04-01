import os
from dotenv import load_dotenv
import anthropic
from retriever import retrieve

# Load from .env for local development.
# On Streamlit Cloud, secrets are injected via st.secrets instead.
load_dotenv()


def get_api_key():
    """
    Get the Anthropic API key from whichever source is available.
    - Locally: loaded from .env file via load_dotenv()
    - Streamlit Cloud: injected via st.secrets

    We try st.secrets first so Streamlit Cloud always takes precedence.
    If streamlit isn't installed or secrets aren't set, we fall back to
    the environment variable set by load_dotenv().
    """
    try:
        import streamlit as st
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.getenv("ANTHROPIC_API_KEY")


# Initialize the Anthropic client once at module level.
client = anthropic.Anthropic(api_key=get_api_key())

# The Claude model to use for generating answers.
# claude-3-5-haiku is fast and cheap — good for a personal chatbot.
# You can switch to claude-opus-4-6 later for higher quality answers.
MODEL = "claude-haiku-4-5-20251001"


def build_prompt(query, chunks):
    """
    Assemble the prompt we send to Claude.

    We structure it as:
    1. A system message that tells Claude its role and rules
    2. The retrieved chunks as "context"
    3. The user's question

    Why pass chunks as context? Claude doesn't know about Lenny's content —
    it was never trained on it. By pasting the relevant chunks directly into
    the prompt, we give Claude the information it needs to answer accurately.
    This is the "augmented" part of Retrieval-Augmented Generation.
    """
    # Format each chunk with its source so Claude can cite it
    context_blocks = []
    for i, chunk in enumerate(chunks):
        context_blocks.append(
            f"[Source {i+1}: {chunk['title']} ({chunk['type']}, {chunk['date']})]\n{chunk['text']}"
        )

    context = "\n\n".join(context_blocks)

    system_prompt = """You are a helpful assistant that answers questions based on content
from Lenny Rachitsky's newsletters and podcast transcripts.

Rules:
- Only answer based on the provided context. Do not use outside knowledge.
- If the context doesn't contain enough information to answer, say so clearly.
- Keep answers concise and grounded in what the sources actually say.
- Reference the source titles when relevant (e.g. "According to the podcast with Albert Cheng...")"""

    user_message = f"""Context:
{context}

Question: {query}"""

    return system_prompt, user_message


def generate_answer(query):
    """
    Full RAG pipeline for a single query:
    1. Retrieve relevant chunks
    2. Build a prompt with those chunks as context
    3. Send to Claude and return the answer
    """
    # Step 1: retrieve relevant chunks
    chunks = retrieve(query)

    if not chunks:
        return "No relevant content found for your question.", []

    # Step 2: build the prompt
    system_prompt, user_message = build_prompt(query, chunks)

    # Step 3: call Claude
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    answer = response.content[0].text
    return answer, chunks


# Test the full end-to-end pipeline
if __name__ == "__main__":
    query = "How should a PM think about product growth?"

    print(f"Question: {query}\n")
    print("Retrieving and generating answer...\n")

    answer, chunks = generate_answer(query)

    print("=== Answer ===")
    print(answer)

    print("\n=== Sources Used ===")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk['title']} ({chunk['type']}, {chunk['date']}) — similarity: {chunk['similarity']}")
