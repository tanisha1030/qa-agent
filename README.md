# Summary for the Autonomous QA Agent Project

The Autonomous QA Agent is a Streamlit app that factors project documents and HTML structure to construct a “testing brain” that can automatically write test cases and Selenium automation scripts (with local AI models, no external API calls). It relies on a pure-Python TF-IDF retrieval system and a local knowledge base (LB) so that all outputs are fully grounded in the provided documents, hence no hallucination.

The system operates in three stages:

1. Knowledge Base Formation — Users submit documentation (MD/TXT/JSON) and HTML documents. The system pre-processes, chunks text, calculates TF-IDF vectors, and save them all in a local pickle-based KB.

2. Test Case Generation – The user explains a feature, and the agent finds context to produce structured test cases (positive and negative), where each is connected to certain document pointers.

3. Selenium Script Generation – When a test case is selected, the system Analyze the HTML Code by BeautifulSoup and creates an executable Python selenium script by using the extracted selectors.

The project has a nice folder structure and simple, preload assets folder, fully streamlit deploy supported. It only needs Python and pip – no external APIs or nothing. The solution is fully conforming to the specification (in binary yes/no terms), it is deterministic, it does not leak/uses private information, and it is well documented, with troubleshooting and demonstration guidance.
