import os
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Supermemory SDK is optional; app runs even if it's not installed.
try:
    from supermemory import Supermemory
except Exception:
    Supermemory = None


# ---------- Models ----------
@st.cache_resource
def load_models():
    # Lightweight + fast
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    gen_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    return embed_model, gen_pipeline


# ---------- Supermemory client (auto-detect) ----------
@st.cache_resource
def init_supermemory():
    """
    Tries in this order:
      1) Streamlit Cloud Secrets (st.secrets["SUPERMEMORY_API_KEY"])
      2) Environment variable (SUPERMEMORY_API_KEY)
    Returns a Supermemory client or None (graceful fallback).
    """
    if Supermemory is None:
        return None

    api_key = None
    # 1) Streamlit Secrets
    try:
        if "SUPERMEMORY_API_KEY" in st.secrets:
            api_key = st.secrets["SUPERMEMORY_API_KEY"]
    except Exception:
        pass

    # 2) Env var
    if not api_key:
        api_key = os.environ.get("SUPERMEMORY_API_KEY")

    if not api_key:
        return None

    try:
        return Supermemory(api_key=api_key, base_url="https://api.supermemory.ai/")
    except Exception:
        return None


embed_model, gen_pipeline = load_models()
sm_client = init_supermemory()


# ---------- Local knowledge base ----------
DOCS = [
    "The next trillion-dollar company will likely dominate an industry where AI agents become indispensable, integrating deeply into workflows and consumer life.",
    "Sustainability and climate-tech may fuel the rise of the next trillion-dollar firm, driven by breakthroughs in clean energy, carbon capture, and scalable green infrastructure.",
    "The next trillion-dollar opportunity could come from biotech, where personalized medicine and gene editing converge with AI to drastically lower costs and improve outcomes.",
    "Platforms that orchestrate AI-first enterprises—where every department is agentic and self-improving—are strong contenders for trillion-dollar scale.",
    "Space economy expansion, including satellite networks, asteroid mining, and interplanetary logistics, could create trillion-dollar valuations within two decades.",
    "Trust and security will be non-negotiable; the next trillion-dollar company will win by proving it can deploy AI at scale safely, ethically, and under tight regulatory frameworks.",
    "Consumer adoption will hinge on seamless design; trillion-dollar winners will make advanced AI invisible and intuitive, like the iPhone did for mobile computing.",
]


# ---------- Build FAISS once ----------
@st.cache_resource
def build_index(texts):
    embs = embed_model.encode(texts, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)
    return index

INDEX = build_index(DOCS)


# ---------- Retrieval helpers ----------
def supermemory_search(question: str, k: int = 2):
    """Return up to k text snippets from Supermemory. Gracefully handles errors."""
    if sm_client is None:
        return []
    try:
        resp = sm_client.search.execute(q=question, top_k=k)
        out = []
        for r in getattr(resp, "results", []):
            # Best-effort extraction of text field names
            text = (
                (r.get("text") if isinstance(r, dict) else None)
                or (r.get("content") if isinstance(r, dict) else None)
                or (r.get("snippet") if isinstance(r, dict) else None)
                or (str(r) if r else None)
            )
            if text:
                out.append(text.strip())
        return out
    except Exception as e:
        # Do not crash the app if Supermemory is down/misconfigured
        return [f"(Supermemory unavailable: {e})"]


def retrieve_context(question: str, k_local: int = 3, k_sm: int = 2):
    # Local top-k from FAISS
    q_emb = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    D, I = INDEX.search(q_emb, k_local)
    local_hits = [DOCS[i] for i in I[0] if 0 <= i < len(DOCS)]

    # Optional Supermemory
    sm_hits = supermemory_search(question, k=k_sm)

    # De-duplicate while preserving order
    seen, combined = set(), []
    for t in (local_hits + sm_hits):
        if not t:
            continue
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            combined.append(t)

    # Keep it lightweight
    return combined[:5]


# ---------- Answer generation ----------
def generate_answer(question: str):
    context_snips = retrieve_context(question)
    context_block = "\n- " + "\n- ".join(context_snips) if context_snips else " (no external context) "

    prompt = f"""You are Angel, a concise product strategist.
Use the following context snippets to ground your answer. If the context is weak, use general knowledge but keep it plausible and concrete. Avoid repetition.

Context:{context_block}

Question: {question}
Answer in 2-4 crisp sentences:
"""

    result = gen_pipeline(
        prompt,
        max_new_tokens=180,
        do_sample=False,
        truncation=True,
    )
    text = result[0]["generated_text"]
    return text, context_snips


# ---------- UI ----------
st.title("Angel")
st.caption("RAG with local notes + optional Supermemory (auto-detected)")

# Optional status expander (collapsed by default)
with st.expander("System status", expanded=False):
    st.write(f"Supermemory: {'connected' if sm_client else 'disabled'}")
    st.write("Models: all-MiniLM-L6-v2 (embeddings), FLAN-T5-base (generation)")

# Chat-style interface (no button; press Enter to submit)
if "history" not in st.session_state:
    st.session_state.history = []

# Render past turns
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

# New user input
user_q = st.chat_input("Ask Angel…")
if user_q:
    # Show user turn
    st.session_state.history.append(("user", user_q))
    with st.chat_message("user"):
        st.markdown(user_q)

    # Generate answer
    answer, ctx = generate_answer(user_q)

    # Show assistant turn
    with st.chat_message("assistant"):
        st.markdown(answer)
        if ctx:
            with st.expander("Context used"):
                for c in ctx:
                    st.write("- ", c)

    st.session_state.history.append(("assistant", answer))
