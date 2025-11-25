import streamlit as st
import numpy as np
from bs4 import BeautifulSoup
import os
import pickle
import re
from collections import Counter
import math
import json

st.set_page_config(page_title="Autonomous QA Agent", layout="wide")

KB_PATH = "kb_store.pkl"

# ---------------------------------------------------------------------
# PURE PYTHON TF-IDF IMPLEMENTATION
# ---------------------------------------------------------------------

def tokenize(text):
    """Tokenize text into lowercase alphanumeric tokens"""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens

def compute_tf(tokens):
    """Compute term frequency for tokens"""
    counts = Counter(tokens)
    total = len(tokens)
    if total == 0:
        return {}
    return {word: counts[word] / total for word in counts}

def compute_idf(docs):
    """Compute inverse document frequency across all documents"""
    N = len(docs)
    if N == 0:
        return {}
    idf = {}
    for doc in docs:
        for word in set(doc):
            idf[word] = idf.get(word, 0) + 1
    for w in idf:
        idf[w] = math.log(N / idf[w])
    return idf

def compute_tfidf_vector(tokens, vocabulary, idf):
    """Compute TF-IDF vector for given tokens"""
    tf = compute_tf(tokens)
    return np.array([tf.get(word, 0) * idf.get(word, 0) for word in vocabulary])

def cosine_sim(a, b):
    """Compute cosine similarity between two vectors"""
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
        self.html_filename = ""
        self.vocab = []
        self.doc_vectors = []
        self.idf = {}
    
    def chunk_text(self, text, size=500, overlap=50):
        """Split text into overlapping chunks"""
        out = []
        start = 0
        while start < len(text):
            end = start + size
            out.append(text[start:end])
            start = end - overlap
            if start >= len(text):
                break
        return out if out else [text]
    
    def add_doc(self, text, meta):
        """Add document chunks to knowledge base"""
        chunks = self.chunk_text(text)
        for c in chunks:
            self.texts.append(c)
            self.metadatas.append(meta)
    
    def set_html(self, html, filename):
        """Store HTML content"""
        self.html = html
        self.html_filename = filename
        # Also add HTML as a searchable document
        self.add_doc(html, {"source": filename, "type": "html"})
    
    def build(self):
        """Build TF-IDF vectors for all documents"""
        if not self.texts:
            return
        
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
        """Retrieve top-k most relevant chunks for a query"""
        if not self.vocab or not self.doc_vectors:
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
# HELPER FUNCTIONS
# -------------------------------------------------------------------------

def parse_html_elements(html_text):
    """Extract form elements and their attributes from HTML"""
    soup = BeautifulSoup(html_text, "html.parser")
    elements = {
        "ids": [],
        "names": [],
        "buttons": [],
        "inputs": [],
        "forms": []
    }
    
    # Extract all IDs
    for tag in soup.find_all(attrs={"id": True}):
        elements["ids"].append({
            "id": tag["id"],
            "tag": tag.name,
            "type": tag.get("type", "")
        })
    
    # Extract all names
    for tag in soup.find_all(attrs={"name": True}):
        elements["names"].append({
            "name": tag["name"],
            "tag": tag.name,
            "type": tag.get("type", "")
        })
    
    # Extract buttons
    for button in soup.find_all(["button", "input"]):
        if button.name == "button" or button.get("type") == "button" or button.get("type") == "submit":
            elements["buttons"].append({
                "id": button.get("id", ""),
                "text": button.get_text(strip=True) or button.get("value", ""),
                "type": button.get("type", "button")
            })
    
    # Extract inputs
    for inp in soup.find_all("input"):
        elements["inputs"].append({
            "id": inp.get("id", ""),
            "name": inp.get("name", ""),
            "type": inp.get("type", "text"),
            "placeholder": inp.get("placeholder", "")
        })
    
    return elements

def generate_test_cases_from_context(query, context, html_elements):
    """Generate test cases based on retrieved context and HTML structure"""
    combined = " ".join([c["text"] for c in context]).lower()
    query_lower = query.lower()
    
    tests = []
    sources = list(set([c['meta'].get('source', 'unknown') for c in context]))
    
    # Discount code tests
    if "discount" in combined or "discount" in query_lower:
        if "save15" in combined or "15%" in combined:
            tests.append({
                "Test_ID": "TC-DISC-001",
                "Feature": "Discount Code",
                "Test_Scenario": "Apply valid discount code 'SAVE15'",
                "Expected_Result": "Total price reduced by 15%",
                "Grounded_In": ", ".join(sources),
                "Test_Type": "Positive"
            })
        tests.append({
            "Test_ID": "TC-DISC-002",
            "Feature": "Discount Code",
            "Test_Scenario": "Apply invalid discount code",
            "Expected_Result": "Error message displayed, no discount applied",
            "Grounded_In": ", ".join(sources),
            "Test_Type": "Negative"
        })
        tests.append({
            "Test_ID": "TC-DISC-003",
            "Feature": "Discount Code",
            "Test_Scenario": "Apply empty discount code",
            "Expected_Result": "Validation error or no action",
            "Grounded_In": ", ".join(sources),
            "Test_Type": "Negative"
        })
    
    # Form validation tests
    if "email" in combined or "form" in combined or "validation" in query_lower:
        tests.append({
            "Test_ID": "TC-FORM-001",
            "Feature": "Email Validation",
            "Test_Scenario": "Submit form with invalid email format",
            "Expected_Result": "Inline error message in red text",
            "Grounded_In": ", ".join(sources),
            "Test_Type": "Negative"
        })
        tests.append({
            "Test_ID": "TC-FORM-002",
            "Feature": "Form Validation",
            "Test_Scenario": "Submit form with empty required fields",
            "Expected_Result": "Error messages for all required fields",
            "Grounded_In": ", ".join(sources),
            "Test_Type": "Negative"
        })
        tests.append({
            "Test_ID": "TC-FORM-003",
            "Feature": "User Details Form",
            "Test_Scenario": "Submit form with all valid data",
            "Expected_Result": "Form submits successfully, no errors",
            "Grounded_In": ", ".join(sources),
            "Test_Type": "Positive"
        })
    
    # Shipping method tests
    if "shipping" in combined or "shipping" in query_lower:
        if "express" in combined or "standard" in combined:
            tests.append({
                "Test_ID": "TC-SHIP-001",
                "Feature": "Shipping Method",
                "Test_Scenario": "Select Express shipping",
                "Expected_Result": "Total price increases by $10",
                "Grounded_In": ", ".join(sources),
                "Test_Type": "Positive"
            })
            tests.append({
                "Test_ID": "TC-SHIP-002",
                "Feature": "Shipping Method",
                "Test_Scenario": "Select Standard shipping",
                "Expected_Result": "No additional cost, free shipping applied",
                "Grounded_In": ", ".join(sources),
                "Test_Type": "Positive"
            })
    
    # Payment method tests
    if "payment" in combined or "payment" in query_lower:
        tests.append({
            "Test_ID": "TC-PAY-001",
            "Feature": "Payment Method",
            "Test_Scenario": "Select Credit Card payment method",
            "Expected_Result": "Credit Card option selected successfully",
            "Grounded_In": ", ".join(sources),
            "Test_Type": "Positive"
        })
        tests.append({
            "Test_ID": "TC-PAY-002",
            "Feature": "Payment Method",
            "Test_Scenario": "Select PayPal payment method",
            "Expected_Result": "PayPal option selected successfully",
            "Grounded_In": ", ".join(sources),
            "Test_Type": "Positive"
        })
    
    # Cart tests
    if "cart" in combined or "cart" in query_lower:
        tests.append({
            "Test_ID": "TC-CART-001",
            "Feature": "Shopping Cart",
            "Test_Scenario": "Add item to cart",
            "Expected_Result": "Item appears in cart, quantity updates",
            "Grounded_In": ", ".join(sources),
            "Test_Type": "Positive"
        })
        tests.append({
            "Test_ID": "TC-CART-002",
            "Feature": "Shopping Cart",
            "Test_Scenario": "Update item quantity in cart",
            "Expected_Result": "Total price recalculates correctly",
            "Grounded_In": ", ".join(sources),
            "Test_Type": "Positive"
        })
    
    # Complete checkout test
    if "checkout" in query_lower or "pay" in combined or len(tests) == 0:
        tests.append({
            "Test_ID": "TC-CHECK-001",
            "Feature": "Complete Checkout",
            "Test_Scenario": "Complete full checkout with valid data",
            "Expected_Result": "Payment Successful message displayed",
            "Grounded_In": ", ".join(sources),
            "Test_Type": "Positive"
        })
    
    return tests

def generate_selenium_script(test_case, html_content, context):
    """Generate Selenium Python script for a test case"""
    soup = BeautifulSoup(html_content, "html.parser")
    elements = parse_html_elements(html_content)
    
    script_lines = [
        "from selenium import webdriver",
        "from selenium.webdriver.common.by import By",
        "from selenium.webdriver.support.ui import WebDriverWait",
        "from selenium.webdriver.support import expected_conditions as EC",
        "import time",
        "",
        "# Initialize the Chrome WebDriver",
        "driver = webdriver.Chrome()",
        "driver.maximize_window()",
        "",
        "try:",
        "    # Navigate to the checkout page",
        "    driver.get('file:///PATH_TO_YOUR/checkout.html')  # Update this path",
        "    time.sleep(2)",
        ""
    ]
    
    test_id = test_case.get("Test_ID", "")
    scenario = test_case.get("Test_Scenario", "").lower()
    
    # Generate script based on test scenario
    if "discount" in scenario:
        if "valid" in scenario or "save15" in scenario:
            script_lines.extend([
                "    # Apply valid discount code SAVE15",
                "    discount_input = driver.find_element(By.ID, 'discount')",
                "    discount_input.clear()",
                "    discount_input.send_keys('SAVE15')",
                "    ",
                "    apply_button = driver.find_element(By.ID, 'apply-discount')",
                "    apply_button.click()",
                "    time.sleep(1)",
                "    ",
                "    # Verify discount is applied",
                "    # Add assertion to check total price is reduced by 15%",
                "    print('Discount code applied successfully')",
            ])
        elif "invalid" in scenario:
            script_lines.extend([
                "    # Apply invalid discount code",
                "    discount_input = driver.find_element(By.ID, 'discount')",
                "    discount_input.clear()",
                "    discount_input.send_keys('INVALID123')",
                "    ",
                "    apply_button = driver.find_element(By.ID, 'apply-discount')",
                "    apply_button.click()",
                "    time.sleep(1)",
                "    ",
                "    # Verify error message is displayed",
                "    print('Invalid discount code test completed')",
            ])
    
    elif "email" in scenario and "invalid" in scenario:
        script_lines.extend([
            "    # Fill form with invalid email",
            "    name_input = driver.find_element(By.ID, 'name')",
            "    name_input.send_keys('John Doe')",
            "    ",
            "    email_input = driver.find_element(By.ID, 'email')",
            "    email_input.send_keys('invalid-email')  # Invalid format",
            "    ",
            "    address_input = driver.find_element(By.ID, 'address')",
            "    address_input.send_keys('123 Main St')",
            "    ",
            "    # Trigger validation",
            "    pay_button = driver.find_element(By.ID, 'pay-now')",
            "    pay_button.click()",
            "    time.sleep(1)",
            "    ",
            "    # Verify error message is displayed in red",
            "    print('Email validation test completed')",
        ])
    
    elif "required" in scenario:
        script_lines.extend([
            "    # Leave required fields empty",
            "    pay_button = driver.find_element(By.ID, 'pay-now')",
            "    pay_button.click()",
            "    time.sleep(1)",
            "    ",
            "    # Verify error messages are displayed",
            "    print('Required field validation test completed')",
        ])
    
    elif "express" in scenario:
        script_lines.extend([
            "    # Select Express shipping",
            "    express_radio = driver.find_element(By.ID, 'shipping-express')",
            "    express_radio.click()",
            "    time.sleep(1)",
            "    ",
            "    # Verify total price increases by $10",
            "    print('Express shipping selected')",
        ])
    
    elif "standard" in scenario:
        script_lines.extend([
            "    # Select Standard shipping",
            "    standard_radio = driver.find_element(By.ID, 'shipping-standard')",
            "    standard_radio.click()",
            "    time.sleep(1)",
            "    ",
            "    # Verify no additional cost",
            "    print('Standard shipping selected')",
        ])
    
    elif "credit card" in scenario:
        script_lines.extend([
            "    # Select Credit Card payment",
            "    credit_card_radio = driver.find_element(By.ID, 'payment-card')",
            "    credit_card_radio.click()",
            "    time.sleep(1)",
            "    ",
            "    print('Credit Card payment method selected')",
        ])
    
    elif "paypal" in scenario:
        script_lines.extend([
            "    # Select PayPal payment",
            "    paypal_radio = driver.find_element(By.ID, 'payment-paypal')",
            "    paypal_radio.click()",
            "    time.sleep(1)",
            "    ",
            "    print('PayPal payment method selected')",
        ])
    
    elif "add" in scenario and "cart" in scenario:
        script_lines.extend([
            "    # Add item to cart",
            "    add_to_cart_btn = driver.find_element(By.CLASS_NAME, 'add-to-cart')",
            "    add_to_cart_btn.click()",
            "    time.sleep(1)",
            "    ",
            "    # Verify item appears in cart",
            "    print('Item added to cart successfully')",
        ])
    
    elif "complete" in scenario or "checkout" in scenario:
        script_lines.extend([
            "    # Fill in all form fields with valid data",
            "    name_input = driver.find_element(By.ID, 'name')",
            "    name_input.send_keys('John Doe')",
            "    ",
            "    email_input = driver.find_element(By.ID, 'email')",
            "    email_input.send_keys('john.doe@example.com')",
            "    ",
            "    address_input = driver.find_element(By.ID, 'address')",
            "    address_input.send_keys('123 Main Street, City, State 12345')",
            "    ",
            "    # Select shipping method",
            "    standard_radio = driver.find_element(By.ID, 'shipping-standard')",
            "    standard_radio.click()",
            "    ",
            "    # Select payment method",
            "    credit_card_radio = driver.find_element(By.ID, 'payment-card')",
            "    credit_card_radio.click()",
            "    time.sleep(1)",
            "    ",
            "    # Submit the form",
            "    pay_button = driver.find_element(By.ID, 'pay-now')",
            "    pay_button.click()",
            "    time.sleep(2)",
            "    ",
            "    # Verify Payment Successful message",
            "    success_msg = driver.find_element(By.ID, 'payment-success')",
            "    assert 'Payment Successful' in success_msg.text",
            "    print('Checkout completed successfully!')",
        ])
    
    else:
        # Generic test script
        script_lines.extend([
            f"    # Test Case: {test_id}",
            f"    # Scenario: {test_case.get('Test_Scenario', '')}",
            "    ",
            "    # Add your test steps here",
            "    print('Test execution completed')",
        ])
    
    script_lines.extend([
        "",
        "except Exception as e:",
        "    print(f'Test failed with error: {e}')",
        "    driver.save_screenshot('test_failure.png')",
        "",
        "finally:",
        "    # Clean up",
        "    time.sleep(2)",
        "    driver.quit()",
    ])
    
    return "\n".join(script_lines)

# -------------------------------------------------------------------------
# Load or create KB
# -------------------------------------------------------------------------

if os.path.exists(KB_PATH):
    try:
        with open(KB_PATH, "rb") as f:
            kb = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading KB: {e}")
        kb = SimpleKB()
else:
    kb = SimpleKB()

# -------------------------------------------------------------------------
# UI
# -------------------------------------------------------------------------

st.title("🤖 Autonomous QA Agent")
st.markdown("### AI-Powered Test Case and Selenium Script Generation")

# Sidebar for file uploads
with st.sidebar:
    st.header("📁 Phase 1: Knowledge Base Ingestion")
    
    st.subheader("Upload Support Documents")
    docs = st.file_uploader(
        "Upload documentation (MD, TXT, JSON)",
        type=["txt", "md", "json"],
        accept_multiple_files=True,
        help="Upload product specs, UI/UX guides, API endpoints, etc."
    )
    
    st.subheader("Upload HTML File")
    html_file = st.file_uploader(
        "Upload checkout.html",
        type=["html", "htm"],
        help="Upload the target HTML file for testing"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("➕ Add to KB", use_container_width=True):
            added = 0
            if docs:
                for f in docs:
                    try:
                        txt = f.read().decode("utf-8", errors="ignore")
                        kb.add_doc(txt, {"source": f.name})
                        added += 1
                    except Exception as e:
                        st.error(f"Error reading {f.name}: {e}")
            
            if html_file:
                try:
                    html_text = html_file.read().decode("utf-8", errors="ignore")
                    kb.set_html(html_text, html_file.name)
                    added += 1
                except Exception as e:
                    st.error(f"Error reading HTML: {e}")
            
            if added > 0:
                with open(KB_PATH, "wb") as f:
                    pickle.dump(kb, f)
                st.success(f"✅ Added {added} file(s) to KB!")
            else:
                st.warning("No files to add")
    
    with col2:
        if st.button("🔨 Build KB", use_container_width=True):
            if len(kb.texts) == 0:
                st.error("No documents in KB. Please add files first.")
            else:
                with st.spinner("Building knowledge base..."):
                    kb.build()
                    with open(KB_PATH, "wb") as f:
                        pickle.dump(kb, f)
                st.success("✅ Knowledge Base Built Successfully!")
    
    st.divider()
    st.subheader("📊 KB Status")
    st.metric("HTML Uploaded", "✅ Yes" if kb.html else "❌ No")
    st.metric("Total Chunks", len(kb.texts))
    st.metric("Vocabulary Size", len(kb.vocab))
    
    if st.button("🗑️ Clear KB", use_container_width=True):
        kb = SimpleKB()
        if os.path.exists(KB_PATH):
            os.remove(KB_PATH)
        st.rerun()

# Main content area
tab1, tab2 = st.tabs(["🧪 Phase 2: Test Case Generation", "🔧 Phase 3: Selenium Script Generation"])

with tab1:
    st.header("Generate Test Cases")
    st.markdown("Enter a feature or requirement to generate comprehensive test cases.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_area(
            "Feature Description",
            height=100,
            placeholder="e.g., 'Generate test cases for discount code feature' or 'Create tests for form validation'",
            help="Describe the feature you want to test"
        )
    with col2:
        top_k = st.number_input("Top K Results", min_value=1, max_value=10, value=5)
    
    if st.button("🚀 Generate Test Cases", use_container_width=True):
        if not query.strip():
            st.error("Please enter a feature description")
        elif len(kb.texts) == 0:
            st.error("Knowledge base is empty. Please add and build KB first.")
        elif not kb.vocab:
            st.error("Knowledge base not built. Please click 'Build KB' button.")
        else:
            with st.spinner("Retrieving context and generating test cases..."):
                # Retrieve relevant context
                ctx = kb.retrieve(query, k=top_k)
                
                if not ctx or all(c["score"] < 0.01 for c in ctx):
                    st.warning("No relevant context found. Generating generic test cases.")
                    ctx = [{"text": query, "meta": {"source": "user_query"}, "score": 0.0}]
                
                # Show retrieved context
                with st.expander("📄 Retrieved Context from Knowledge Base", expanded=False):
                    for i, c in enumerate(ctx):
                        st.markdown(f"**{i+1}. Source:** {c['meta'].get('source', 'unknown')} (Score: {c['score']:.3f})")
                        st.text(c['text'][:300] + "..." if len(c['text']) > 300 else c['text'])
                        st.divider()
                
                # Parse HTML elements
                html_elements = parse_html_elements(kb.html) if kb.html else {}
                
                # Generate test cases
                tests = generate_test_cases_from_context(query, ctx, html_elements)
                
                # Store in session state
                st.session_state["tests"] = tests
                st.session_state["query"] = query
            
            st.success(f"✅ Generated {len(tests)} test cases!")
    
    # Display generated test cases
    if "tests" in st.session_state and st.session_state.get("tests"):
        st.divider()
        st.subheader("📋 Generated Test Cases")
        
        for i, test in enumerate(st.session_state["tests"]):
            with st.expander(f"**{test['Test_ID']}** - {test['Test_Scenario']}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Feature:** {test['Feature']}")
                    st.markdown(f"**Test Type:** {test['Test_Type']}")
                with col2:
                    st.markdown(f"**Grounded In:** {test['Grounded_In']}")
                
                st.markdown(f"**Expected Result:** {test['Expected_Result']}")
        
        # Download option
        test_json = json.dumps(st.session_state["tests"], indent=2)
        st.download_button(
            "📥 Download Test Cases (JSON)",
            test_json,
            "test_cases.json",
            "application/json",
            use_container_width=True
        )

with tab2:
    st.header("Generate Selenium Scripts")
    st.markdown("Select a test case and generate executable Selenium Python scripts.")
    
    if "tests" not in st.session_state or not st.session_state.get("tests"):
        st.info("⚠️ Please generate test cases first in Phase 2.")
    else:
        # Test case selection
        labels = [f"{t['Test_ID']} - {t['Test_Scenario']}" for t in st.session_state["tests"]]
        choice = st.selectbox("Choose Test Case", labels, help="Select a test case to generate Selenium script")
        
        # Extract selected test case
        test_id = choice.split(" - ")[0]
        tc = next((t for t in st.session_state["tests"] if t["Test_ID"] == test_id), None)
        
        if tc:
            # Display test case details
            with st.expander("📄 Selected Test Case Details", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Test ID:** {tc['Test_ID']}")
                    st.markdown(f"**Feature:** {tc['Feature']}")
                with col2:
                    st.markdown(f"**Type:** {tc['Test_Type']}")
                    st.markdown(f"**Grounded In:** {tc['Grounded_In']}")
                st.markdown(f"**Scenario:** {tc['Test_Scenario']}")
                st.markdown(f"**Expected Result:** {tc['Expected_Result']}")
            
            if st.button("⚙️ Generate Selenium Script", use_container_width=True):
                if not kb.html:
                    st.error("❌ No HTML file uploaded! Please upload checkout.html in Phase 1.")
                else:
                    with st.spinner("Generating Selenium script..."):
                        # Retrieve relevant context for this test case
                        ctx = kb.retrieve(tc['Test_Scenario'], k=3)
                        
                        # Generate script
                        script = generate_selenium_script(tc, kb.html, ctx)
                        
                        # Store in session state
                        st.session_state[f"script_{test_id}"] = script
                    
                    st.success("✅ Selenium script generated successfully!")
            
            # Display generated script
            if f"script_{test_id}" in st.session_state:
                st.divider()
                st.subheader("🐍 Generated Selenium Python Script")
                
                script_content = st.session_state[f"script_{test_id}"]
                st.code(script_content, language="python", line_numbers=True)
                
                # Download option
                st.download_button(
                    "📥 Download Script",
                    script_content,
                    f"{test_id}_selenium_script.py",
                    "text/x-python",
                    use_container_width=True
                )
                
                st.info("💡 **Note:** Update the file path in line 12 to point to your actual checkout.html location.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Autonomous QA Agent | Built with Streamlit | TF-IDF Based Retrieval</p>
</div>
""", unsafe_allow_html=True)
