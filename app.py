from dotenv import load_dotenv
load_dotenv()

import os, pickle
import faiss, numpy as np
import streamlit as st
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url="https://integrate.api.nvidia.com/v1"
)

@st.cache_resource
def load_index():
    index = faiss.read_index("notes.index")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def embed_query(query):
    response = client.embeddings.create(
        model="nvidia/nv-embedqa-e5-v5",
        input=query,
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "END"}
    )
    return np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)

def retrieve(query, index, chunks, k=4):
    vec = embed_query(query)
    _, ids = index.search(vec, k)
    return [chunks[i] for i in ids[0]]

def generate_answer(query, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have that in my notes."

Context:
{context}

Question: {query}
Answer:"""
    response = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

st.title("📚 AAC Chatbot")
st.caption("Ask me anything from the course notes!")

index, chunks = load_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching notes..."):
            context = retrieve(query, index, chunks)
            answer = generate_answer(query, context)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})