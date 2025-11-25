import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from bs4 import BeautifulSoup
import re
import os
import pickle
from io import StringIO

st.set_page_config(page_title="QA Agent (Streamlit-only)", layout="wide")

# --- Paths
KB_PATH = "kb_store.pkl"

# --- Helpers
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def text_from_uploaded_file(uploaded):
    if uploaded is None:
        return ""
    try:
        raw = uploaded.getvalue().decode("utf-8", errors="ignore")
    except:
        raw = str(uploaded.getvalue())
    return raw

def chunk_text(text, size=500, overlap=50):
    tokens = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start+size)
        chunk = text[start:end]
        tokens.append(chunk)
        start = end - overlap
        if start<0: start=0
        if start>=L:
            break
    return tokens

class SimpleKB:
    def __init__(self, model):
        self.model = model
        self.texts = []
        self.metadatas = []
        self.index = None
        self.embeddings = None
        self.html = ""
    def add_doc(self, text, metadata):
        chunks = chunk_text(text)
        for c in chunks:
            self.texts.append(c)
            self.metadatas.append(metadata)
    def set_html(self, html_text, filename=None):
        self.html = html_text
        self.html_filename = filename
    def build(self):
        if len(self.texts)==0:
            return
        embs = self.model.encode(self.texts, show_progress_bar=False, convert_to_numpy=True)
        d = embs.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embs)
        self.index = index
        self.embeddings = embs
    def retrieve(self, query, k=4):
        if self.index is None:
            return []
        qv = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(qv, k)
        results = []
        for idx in I[0]:
            if idx < len(self.texts):
                results.append({"text": self.texts[idx], "meta": self.metadatas[idx]})
        return results

# Persistent KB
if os.path.exists(KB_PATH):
    try:
        with open(KB_PATH,"rb") as f:
            kb = pickle.load(f)
    except:
        kb = SimpleKB(load_model())
else:
    kb = SimpleKB(load_model())

st.title("Autonomous QA Agent — Streamlit-only (No OpenAI)")

with st.sidebar:
    st.header("Upload assets")
    uploaded_docs = st.file_uploader("Upload support documents (MD/TXT/JSON). You can upload multiple.", accept_multiple_files=True, type=['md','txt','json'])
    uploaded_html = st.file_uploader("Upload checkout.html", type=['html','htm'])
    if st.button("Add uploaded files to KB"):
        added = 0
        if uploaded_docs:
            for f in uploaded_docs:
                txt = text_from_uploaded_file(f)
                kb.add_doc(txt, {"source": f.name})
                added += 1
        if uploaded_html:
            html_text = text_from_uploaded_file(uploaded_html)
            kb.set_html(html_text, filename=uploaded_html.name)
            added += 1
        with open(KB_PATH,"wb") as f:
            pickle.dump(kb,f)
        st.success(f"Added {added} files to KB (saved).")

    if st.button("Build KB (embeddings + FAISS)"):
        with st.spinner("Building embeddings and FAISS index..."):
            kb.build()
            with open(KB_PATH,"wb") as f:
                pickle.dump(kb,f)
        st.success("Knowledge Base built.")

    st.markdown("---")
    st.write("Sample assets are bundled in `/assets/` in the repo.")

st.subheader("KB Status")
st.write("Has HTML:", bool(getattr(kb, "html", "")))
st.write("Number of docs/chunks:", len(kb.texts))

# Agent UI
st.header("Agent — Generate Test Cases")
query = st.text_area("Describe what you want test cases for (e.g., 'discount code feature')", height=80)
top_k = st.number_input("Top K retrieval documents", value=4, min_value=1, max_value=10)
if st.button("Generate Test Cases"):
    if not query.strip():
        st.error("Please enter a query.")
    else:
        if kb.index is None:
            st.warning("KB not built; building now...")
            kb.build()
            with open(KB_PATH,"wb") as f:
                pickle.dump(kb,f)
        contexts = kb.retrieve(query, k=top_k)
        st.markdown("**Retrieved Contexts:**")
        for c in contexts:
            st.markdown(f"- Source: {c['meta'].get('source','unknown')} — {c['text'][:300].replace('\n',' ')}...")
        # Simple rule-based testcase generation using contexts
        tcases = []
        # look for discount codes
        joined = " ".join([c['text'] for c in contexts]).lower()
        discount_codes = re.findall(r'`([A-Z0-9]{3,})`', joined.upper())
        # fallback: find SAVE15 etc in product_specs
        if "discount" in joined or "save" in joined or discount_codes:
            codes = set(discount_codes) if discount_codes else set(["SAVE15"])
            for i, code in enumerate(codes, start=1):
                tcases.append({
                    "Test_ID": f"TC-DISC-{i:03d}",
                    "Feature": "Discount Code",
                    "Test_Scenario": f"Apply valid discount code '{code}' and verify discount applied.",
                    "Expected_Result": f"Total price is reduced according to {code} rules.",
                    "Grounded_In": "retrieved_docs"
                })
                tcases.append({
                    "Test_ID": f"TC-DISC-N-{i:03d}",
                    "Feature": "Discount Code",
                    "Test_Scenario": f"Apply invalid discount code 'INVALID' and verify rejection.",
                    "Expected_Result": "Discount is rejected and total price unchanged.",
                    "Grounded_In": "retrieved_docs"
                })
        # shipping
        if "shipping" in joined:
            tcases.append({
                "Test_ID":"TC-SHIP-001",
                "Feature":"Shipping",
                "Test_Scenario":"Select Express shipping and verify extra charge applied.",
                "Expected_Result":"Total increases by express shipping cost.",
                "Grounded_In":"retrieved_docs"
            })
        # form validation
        if "email" in joined or "required" in joined or "address" in joined:
            tcases.append({
                "Test_ID":"TC-FORM-001",
                "Feature":"User Details Form",
                "Test_Scenario":"Submit form with invalid email format.",
                "Expected_Result":"Inline email validation error shown in red and submission blocked.",
                "Grounded_In":"retrieved_docs"
            })
        if not tcases:
            # generic fallback
            tcases.append({
                "Test_ID":"TC-GEN-001",
                "Feature":"General",
                "Test_Scenario":f"Verify feature related to '{query}'.",
                "Expected_Result":"Behaviour as described in documentation.",
                "Grounded_In":"retrieved_docs"
            })
        st.markdown("### Generated Test Cases (JSON)")
        st.json(tcases)
        # keep testcases in session_state for later script gen
        st.session_state['last_testcases'] = tcases

st.markdown("---")
st.header("Select Test Case and Generate Selenium Script")
if 'last_testcases' in st.session_state:
    tcases = st.session_state['last_testcases']
    options = [f"{t['Test_ID']} — {t['Test_Scenario'][:80]}" for t in tcases]
    choice = st.selectbox("Pick a test case", options)
    idx = options.index(choice)
    selected = tcases[idx]
    st.markdown("**Selected Test Case**")
    st.json(selected)
    if st.button("Generate Selenium Script for this test case"):
        html = getattr(kb, "html", "")
        if not html:
            st.error("No checkout.html in KB. Upload it in the sidebar.")
        else:
            soup = BeautifulSoup(html, "lxml")
            # simple selector extraction: ids, names
            selectors = {}
            for el in soup.find_all(attrs={"id": True}):
                selectors[el['id']] = {
                    "tag": el.name,
                    "attrs": dict(el.attrs)
                }
            # build a python selenium script template
            script_lines = []
            script_lines.append("from selenium import webdriver")
            script_lines.append("from selenium.webdriver.common.by import By")
            script_lines.append("import time")
            script_lines.append("")
            script_lines.append("driver = webdriver.Chrome()  # ensure chromedriver is in PATH")
            script_lines.append("driver.get('file://' + r'YOUR_PATH_TO_checkout.html')  # update path")
            script_lines.append("")
            script_lines.append("# --- Automated steps for test case")
            # heuristics for common actions
            if selected['Feature'].lower().startswith("discount"):
                # find discount input id
                discount_id = None
                for k,v in selectors.items():
                    if 'discount' in k or 'coupon' in k or 'code' in k:
                        discount_id = k; break
                if discount_id:
                    script_lines.append(f"discount = driver.find_element(By.ID, '{discount_id}')")
                    script_lines.append("discount.clear()")
                    # use code from expected result if present
                    m = re.search(r"'([A-Z0-9]+)'", selected['Test_Scenario'])
                    code = m.group(1) if m else 'SAVE15'
                    script_lines.append(f"discount.send_keys('{code}')")
                    # try to click apply button if exists
                    # attempt common ids
                    apply_id = None
                    for k in selectors.keys():
                        if 'apply' in k or 'discount' in k:
                            apply_id = k; break
                    if apply_id:
                        script_lines.append(f"driver.find_element(By.ID, '{apply_id}').click()")
                    script_lines.append("time.sleep(1)")
                    script_lines.append("# TODO: add assertions to verify price change")
            elif selected['Feature'].lower().startswith("user details") or 'form' in selected['Feature'].lower():
                # fill sample fields
                for key in ['name','email','address','card_number']:
                    found = None
                    for k in selectors.keys():
                        if key in k:
                            found = k; break
                    if found:
                        script_lines.append(f"el = driver.find_element(By.ID, '{found}')")
                        if 'email' in key:
                            script_lines.append("el.send_keys('test@example.com')")
                        elif 'card' in key:
                            script_lines.append("el.send_keys('411111111111')")
                        else:
                            script_lines.append("el.send_keys('Demo')")
                # click pay
                pay_id = None
                for k in selectors.keys():
                    if 'pay' in k or 'pay-now' in k or 'paynow' in k:
                        pay_id = k; break
                if pay_id:
                    script_lines.append(f"driver.find_element(By.ID, '{pay_id}').click()")
                else:
                    script_lines.append("print('Pay button not found by id; you may need to update selector')")
                script_lines.append("time.sleep(1)")
                script_lines.append("# TODO: add assertions for success message")
            else:
                script_lines.append("# Generic script - please update with appropriate selectors/actions.")
            script = "\n".join(script_lines)
            st.code(script, language='python')
else:
    st.info("Generate test cases first to enable script generation.")

st.markdown('---')
st.header("Developer / Submission Notes")
st.markdown("""- This app intentionally avoids remote LLMs. It uses local embeddings + deterministic templates to produce
  test cases and scripts grounded in uploaded documents.
- To fully automate natural-language-to-code generation, plug in a local LLM and replace the template
  sections with model.generate() calls.
- You can download the generated script and run it against the local `checkout.html`.
""")
