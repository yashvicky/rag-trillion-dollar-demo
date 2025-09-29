import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline

# --- Load models ---
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    extractive_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    generative_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    return embed_model, extractive_pipeline, generative_pipeline

embed_model, extractive_pipeline, generative_pipeline = load_models()

# --- Knowledge base: your notes ---
docs = [
    "The next trillion-dollar company will likely dominate an industry where AI agents become indispensable, integrating deeply into workflows and consumer life.",
    "Sustainability and climate-tech may fuel the rise of the next trillion-dollar firm, driven by breakthroughs in clean energy, carbon capture, and scalable green infrastructure.",
    "The next trillion-dollar opportunity could come from biotech, where personalized medicine and gene editing converge with AI to drastically lower costs and improve outcomes.",
    "Platforms that orchestrate AI-first enterprises—where every department is agentic and self-improving—are strong contenders for trillion-dollar scale.",
    "Space economy expansion, including satellite networks, asteroid mining, and interplanetary logistics, could create trillion-dollar valuations within two decades.",
    "Trust and security will be non-negotiable; the next trillion-dollar company will win by proving it can deploy AI at scale safely, ethically, and under tight regulatory frameworks.",
    "Consumer adoption will hinge on seamless design; trillion-dollar winners will make advanced AI invisible and intuitive, like the iPhone did for mobile computing."
]

# --- Build FAISS index ---
embeddings = embed_model.encode(docs, convert_to_numpy=True).astype("float32")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def rag_extractive(question, k=3):
    q_embed = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_embed, k=k)
    retrieved = [docs[i] for i in I[0]]

    context = " ".join(retrieved)
    result = extractive_pipeline(question=question, context=context)
    return result["answer"], retrieved

def rag_generative(question, k=3):
    q_embed = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_embed, k=k)
    retrieved = [docs[i] for i in I[0]]

    context = " ".join(retrieved)
    prompt = f"""Use the context below to answer concisely and avoid repetition.

Context: {context}

Question: {question}
Answer in 2-3 clear sentences:
"""
    result = generative_pipeline(prompt, max_new_tokens=150, do_sample=False, truncation=True)
    return result[0]["generated_text"], retrieved

# --- Streamlit UI ---
st.title("Challenge 2 - RAG Demo (Trillion-Dollar Company)")

mode = st.radio("Choose QA Mode:", ["Extractive (DistilBERT)", "Generative (Flan-T5)"])
user_q = st.text_input("💡 Ask a question:")

if st.button("Generate Answer", key="ask_button"):
    if user_q.strip():
        if mode == "Extractive (DistilBERT)":
            answer, retrieved = rag_extractive(user_q)
        else:
            answer, retrieved = rag_generative(user_q)

        st.markdown("### 💡 Answer:")
        st.write(answer)

        with st.expander("📚 Context Used"):
            for r in retrieved:
                st.write("-", r)
