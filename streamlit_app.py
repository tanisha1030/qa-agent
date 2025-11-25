import streamlit as st
import numpy as np
from bs4 import BeautifulSoup
import os
import pickle
import re
from collections import Counter
import math

st.set_page_config(page_title="QA Agent (Pure Python TF-IDF)", layout="wide")

KB_PATH = "kb_store.pkl"

# ---------------------------------------------------------------------
# PURE PYTHON TF-IDF IMPLEMENTATION
# ---------------------------------------------------------------------

def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens

def compute_tf(tokens):
    counts = Counter(tokens)
    total = len(tokens)
    return {word: counts[word] / total for word in counts}

def compute_idf(docs):
    N = len(docs)
    idf = {}
    for doc in docs:
        for word in set(doc):
            idf[word] = idf.get(word, 0) + 1
    for w in idf:
        idf[w] = math.log(N / idf[w])
    return idf

def compute_tfidf_vector(tokens, vocabulary, idf):
    tf = compute_tf(tokens)
    return np.array([tf.get(word, 0) * idf.get(word, 0) for word in vocabulary])

def cosine_sim(a, b):
    num = np.dot(a, b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0: 
        return 0
    return num / den

# ---------------------------------------------------------------------
# KB Class (Uses pure python TF-IDF)
# ---------------------------------------------------------------------

class SimpleKB:
    def __init__(self):
        self.texts = []
        self.metadatas = []
        self.html = ""
        self.vocab = []
        self.doc_vectors = []
        self.idf = {}

    def chunk_text(self, text, size=500, overlap=50):
        out = []
        start = 0
        while start < len(text):
            end = start + size
            out.append(text[start:end])
            start = end - overlap
            if start < 0: start = 0
        return out

    def add_doc(self, text, meta):
        for c in self.chunk_text(text):
            self.texts.append(c)
            self.metadatas.append(meta)

    def set_html(self, html, filename):
        self.html = html
        self.html_filename = filename

    def build(self):
        tokenized_docs = [tokenize(t) for t in self.texts]

        # Build vocabulary
        vocab = set()
        for toks in tokenized_docs:
            vocab.update(toks)
        self.vocab = sorted(list(vocab))

        # Compute IDF
        self.idf = compute_idf(tokenized_docs)

        # Build vectors
        self.doc_vectors = [
            compute_tfidf_vector(toks, self.vocab, self.idf) 
            for toks in tokenized_docs
        ]

    def retrieve(self, query, k=4):
        if not self.vocab:
            return []

        q_tokens = tokenize(query)
        q_vec = compute_tfidf_vector(q_tokens, self.vocab, self.idf)

        sims = []
        for i, vec in enumerate(self.doc_vectors):
            sims.append((cosine_sim(q_vec, vec), i))

        sims.sort(reverse=True)

        results = []
        for score, idx in sims[:k]:
            results.append({
                "text": self.texts[idx],
                "meta": self.metadatas[idx],
                "score": float(score)
            })
        return results

# -------------------------------------------------------------------------

# Load or create KB
if os.path.exists(KB_PATH):
    try:
        with open(KB_PATH, "rb") as f:
            kb = pickle.load(f)
    except:
        kb = SimpleKB()
else:
    kb = SimpleKB()

# -------------------------------------------------------------------------
# UI
# -------------------------------------------------------------------------

st.title("Autonomous QA Agent — Pure Python Version (No sklearn, No FAISS, No API)")

with st.sidebar:
    st.header("Upload Files")

    docs = st.file_uploader("Support Docs", type=["txt", "md", "json"], accept_multiple_files=True)
    html_file = st.file_uploader("Checkout HTML", type=["html", "htm"])

    if st.button("Add to KB"):
        added = 0
        if docs:
            for f in docs:
                txt = f.read().decode("utf-8", errors="ignore")
                kb.add_doc(txt, {"source": f.name})
                added += 1
        if html_file:
            html_text = html_file.read().decode("utf-8", errors="ignore")
            kb.set_html(html_text, html_file.name)
            added += 1

        with open(KB_PATH, "wb") as f:
            pickle.dump(kb, f)

        st.success(f"Added {added} files!")

    if st.button("Build KB"):
        kb.build()
        with open(KB_PATH, "wb") as f:
            pickle.dump(kb, f)
        st.success("KB built successfully!")

st.subheader("Status")
st.write("HTML Uploaded:", bool(kb.html))
st.write("Chunks:", len(kb.texts))

# -------------------------------------------------------------------------
# TEST CASE GENERATION
# -------------------------------------------------------------------------

st.header("Generate Test Cases")

query = st.text_area("Enter feature:", height=60)
top_k = st.number_input("Top K", 1, 10, 4)

if st.button("Generate Test Cases"):
    ctx = kb.retrieve(query, k=top_k)

    st.markdown("### Retrieved Context")
    for c in ctx:
        st.write(f"- **{c['meta'].get('source')}** → {c['text'][:150]}...")

    combined = " ".join([c["text"] for c in ctx]).lower()

    tests = []

    if "discount" in combined:
        tests.append({
            "Test_ID": "TC-DISC-001",
            "Scenario": "Apply valid discount code",
            "Expected": "Correct discount applied",
        })
        tests.append({
            "Test_ID": "TC-DISC-002",
            "Scenario": "Apply invalid discount code",
            "Expected": "Discount rejected",
        })

    if "email" in combined:
        tests.append({
            "Test_ID": "TC-FORM-001",
            "Scenario": "Submit invalid email",
            "Expected": "Inline error message",
        })

    if not tests:
        tests.append({
            "Test_ID": "TC-GEN-001",
            "Scenario": f"Test {query}",
            "Expected": "Matches documentation",
        })

    st.json(tests)
    st.session_state["tests"] = tests

# -------------------------------------------------------------------------
# SELENIUM SCRIPT GENERATION
# -------------------------------------------------------------------------

st.header("Generate Selenium Script")

if "tests" in st.session_state:
    labels = [t["Test_ID"] for t in st.session_state["tests"]]
    choice = st.selectbox("Choose Test Case", labels)

    tc = next(t for t in st.session_state["tests"] if t["Test_ID"] == choice)

    if st.button("Generate Script"):
        if not kb.html:
            st.error("No HTML uploaded!")
        else:
            soup = BeautifulSoup(kb.html, "lxml")
            ids = [tag["id"] for tag in soup.find_all(attrs={"id": True})]

            script = [
                "from selenium import webdriver",
                "from selenium.webdriver.common.by import By",
                "import time\n",
                "driver = webdriver.Chrome()",
                "driver.get('file://PATH_TO_checkout.html')\n"
            ]

            if "discount" in tc["Scenario"].lower():
                script += [
                    "driver.find_element(By.ID,'discount').send_keys('SAVE15')",
                    "driver.find_element(By.ID,'apply-discount').click()"
                ]
            else:
                script.append("# Fill common fields")
                for key in ["name", "email", "address"]:
                    for i in ids:
                        if key in i:
                            script.append(f"driver.find_element(By.ID,'{i}').send_keys('test')")

            st.code("\n".join(script), language="python")

else:
    st.info("Generate test cases first.")
