import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import os
import pickle
import re

st.set_page_config(page_title="QA Agent (TF-IDF)", layout="wide")

KB_PATH = "kb_store.pkl"

# ------------- KB Class ----------------
class SimpleKB:
    def __init__(self):
        self.texts = []
        self.metadatas = []
        self.html = ""
        self.html_filename = None

        # TF-IDF
        self.vectorizer = None
        self.doc_vectors = None

    def add_doc(self, text, meta):
        chunks = self.chunk_text(text)
        for c in chunks:
            self.texts.append(c)
            self.metadatas.append(meta)

    def set_html(self, html, filename):
        self.html = html
        self.html_filename = filename

    def chunk_text(self, text, size=500, overlap=50):
        out = []
        start = 0
        while start < len(text):
            end = start + size
            out.append(text[start:end])
            start = end - overlap
            if start < 0:
                start = 0
        return out

    def build(self):
        if not self.texts:
            return
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, k=4):
        if self.vectorizer is None:
            return []

        qv = self.vectorizer.transform([query])
        sim = cosine_similarity(qv, self.doc_vectors)[0]
        top_idx = np.argsort(sim)[::-1][:k]

        results = []
        for idx in top_idx:
            results.append({"text": self.texts[idx],
                            "meta": self.metadatas[idx]})
        return results


# -------- Load or create KB --------
if os.path.exists(KB_PATH):
    try:
        with open(KB_PATH, "rb") as f:
            kb = pickle.load(f)
    except:
        kb = SimpleKB()
else:
    kb = SimpleKB()

# ---------------------- UI -------------------------

st.title("Autonomous QA Agent — (TF-IDF Version, No FAISS, No OpenAI)")

with st.sidebar:
    st.header("Upload documents")

    docs = st.file_uploader("Upload support docs", type=["txt", "md", "json"], accept_multiple_files=True)
    html_file = st.file_uploader("Upload checkout.html", type=["html", "htm"])

    if st.button("Add to KB"):
        added = 0
        if docs:
            for f in docs:
                text = f.read().decode("utf-8", errors="ignore")
                kb.add_doc(text, {"source": f.name})
                added += 1

        if html_file:
            html_text = html_file.read().decode("utf-8", errors="ignore")
            kb.set_html(html_text, html_file.name)
            added += 1

        with open(KB_PATH, "wb") as f:
            pickle.dump(kb, f)

        st.success(f"Added {added} items to KB")

    if st.button("Build KB"):
        kb.build()
        with open(KB_PATH, "wb") as f:
            pickle.dump(kb, f)
        st.success("KB built successfully!")


st.subheader("KB Status")
st.write("HTML Uploaded:", bool(kb.html))
st.write("Document Chunks:", len(kb.texts))

# ------------- Test Case Generation --------------------

st.header("Generate Test Cases")
query = st.text_area("Enter feature to test (e.g. discount code):")
top_k = st.number_input("Top K retrieval", value=4, min_value=1, max_value=10)

if st.button("Generate Test Cases"):
    if not query:
        st.error("Enter a query first")
    else:
        ctx = kb.retrieve(query, k=top_k)

        st.markdown("### Retrieved Contexts")
        for c in ctx:
            st.markdown(f"**{c['meta'].get('source','?')}**: {c['text'][:200]}...")

        combined = " ".join([c["text"] for c in ctx]).lower()

        tcs = []

        # ------- Rule-based test generation -------
        if "discount" in combined or "save" in combined:
            tcs.append({
                "Test_ID": "TC-DIS-001",
                "Feature": "Discount",
                "Test_Scenario": "Apply valid discount code",
                "Expected_Result": "Correct discount applied",
                "Grounded_In": "retrieved_docs"
            })
            tcs.append({
                "Test_ID": "TC-DIS-002",
                "Feature": "Discount",
                "Test_Scenario": "Apply invalid discount code",
                "Expected_Result": "Discount rejected",
                "Grounded_In": "retrieved_docs"
            })

        if "email" in combined or "address" in combined:
            tcs.append({
                "Test_ID": "TC-FORM-001",
                "Feature": "Form Validation",
                "Test_Scenario": "Invalid email",
                "Expected_Result": "Inline red error message",
                "Grounded_In": "retrieved_docs"
            })

        if not tcs:
            tcs.append({
                "Test_ID": "TC-GEN-001",
                "Feature": query,
                "Test_Scenario": "Generic test case",
                "Expected_Result": "Expected behavior from docs",
                "Grounded_In": "retrieved_docs"
            })

        st.json(tcs)
        st.session_state["testcases"] = tcs

# ------------------ Script Generation -----------------------

st.header("Generate Selenium Script")

if "testcases" in st.session_state:
    tc_list = st.session_state["testcases"]
    labels = [f"{tc['Test_ID']} - {tc['Test_Scenario']}" for tc in tc_list]
    choice = st.selectbox("Choose Test Case", labels)

    selected = tc_list[labels.index(choice)]

    if st.button("Generate Script"):
        if not kb.html:
            st.error("No HTML in KB.")
        else:
            soup = BeautifulSoup(kb.html, "lxml")
            ids = {}
            for el in soup.find_all(attrs={"id": True}):
                ids[el["id"]] = el.name

            script = []
            script.append("from selenium import webdriver")
            script.append("from selenium.webdriver.common.by import By")
            script.append("import time\n")
            script.append("driver = webdriver.Chrome()")
            script.append("driver.get('file://PATH_TO_checkout.html')\n")

            if selected["Feature"].lower().startswith("discount"):
                if "discount" in ids:
                    script.append("el = driver.find_element(By.ID,'discount')")
                    script.append("el.send_keys('SAVE15')")
                    script.append("driver.find_element(By.ID,'apply-discount').click()")
                else:
                    script.append("# Discount input not found")
            else:
                script.append("# Fill form fields if available:")
                for key in ["name", "email", "address"]:
                    for i in ids:
                        if key in i:
                            script.append(f"driver.find_element(By.ID,'{i}').send_keys('test')")
                for i in ids:
                    if "pay" in i:
                        script.append(f"driver.find_element(By.ID,'{i}').click()")
                        break

            st.code("\n".join(script), language="python")

else:
    st.info("Generate test cases first.")
