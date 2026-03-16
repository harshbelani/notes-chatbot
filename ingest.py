from dotenv import load_dotenv
load_dotenv()

import os, pickle
from pathlib import Path
from docx import Document
import faiss, numpy as np
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url="https://integrate.api.nvidia.com/v1"
)

def load_docs(folder="notes"):
    texts = []
    for path in Path(folder).glob("*.docx"):
        doc = Document(path)
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        texts.append(full_text)
    return texts

def chunk(text, size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return chunks

def embed(texts):
    response = client.embeddings.create(
        model="nvidia/nv-embedqa-e5-v5",
        input=texts,
        encoding_format="float",
        extra_body={"input_type": "passage", "truncate": "END"}
    )
    return np.array([e.embedding for e in response.data], dtype="float32")

if __name__ == "__main__":
    all_chunks = []
    for doc in load_docs("notes"):
        all_chunks.extend(chunk(doc))

    print(f"Embedding {len(all_chunks)} chunks...")
    embeddings = embed(all_chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, "notes.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print("Done! notes.index and chunks.pkl saved.")