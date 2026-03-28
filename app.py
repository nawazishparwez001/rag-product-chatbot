import streamlit as st
from generator import generate_answer

# --- Page config ---
# This must be the first Streamlit call in the script
st.set_page_config(
    page_title="Lenny's Knowledge Base",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Lenny's Knowledge Base")
st.caption("Ask anything from Lenny's newsletters and podcast transcripts.")

# --- Chat history ---
# st.session_state persists across reruns within the same browser session.
# Without this, the chat history would reset every time the user submits a question.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render all previous messages so the conversation feels continuous
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat input ---
# st.chat_input pins the input box at the bottom of the page.
# It returns the user's text when they hit Enter, otherwise None.
if query := st.chat_input("Ask a question..."):

    # Show the user's message immediately
    with st.chat_message("user"):
        st.markdown(query)

    # Save to history
    st.session_state.messages.append({"role": "user", "content": query})

    # Generate the answer and show a spinner while waiting
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            answer, chunks = generate_answer(query)

        st.markdown(answer)

        # Show the sources in a collapsible section.
        # We show sources so the user can verify where the answer came from —
        # this builds trust and helps spot retrieval errors.
        if chunks:
            with st.expander("📚 Sources used"):
                for i, chunk in enumerate(chunks):
                    st.markdown(
                        f"**{i+1}. {chunk['title']}** ({chunk['type']}, {chunk['date']})  \n"
                        f"Similarity: `{chunk['similarity']}`  \n"
                        f"[View source]({chunk['source_url']})"
                    )

    # Save assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
