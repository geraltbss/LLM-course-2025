from llmsherpa.readers import LayoutPDFReader
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex
from llama_index.core import Document, ServiceContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# Source: https://medium.com/@jitsins/query-complex-pdfs-in-natural-language-with-llmsherpa-ollama-llama3-8b-13b4782243de
# To install:
# 1. run https://stackoverflow.com/questions/52805115/certificate-verify-failed-unable-to-get-local-issuer-certificate
# 2. install and run ollama:
# ollama pull llama3
# ollama run llama3
# 3. Install docker and run:
# docker pull ghcr.io/nlmatics/nlm-ingestor:latest
# docker run -p 5010:5001 ghcr.io/nlmatics/nlm-ingestor:latest
# This will expose the api link “http://localhost:5010/api/parseDocument?renderFormat=all” for you to utilize in your code.

# Initialize LLm
llm = Ollama(model="llama3", request_timeout=60.0)

llmsherpa_api_url = "http://localhost:5010/api/parseDocument?renderFormat=all"
pdf_url = "https://s206.q4cdn.com/479360582/files/doc_financials/2024/q1/2024q1-alphabet-earnings-release-pdf.pdf"
pdf_reader = LayoutPDFReader(llmsherpa_api_url)

# Read PDF
doc = pdf_reader.read_pdf(pdf_url)

# Get data from the Section by Title
table_sections = []

for section in doc.sections():
    # Heuristic: include sections that contain tables or look tabular
    html = section.to_html(include_children=True, recurse=True)
    if "<table" in html.lower():
        table_sections.append(
            f"<h2>{section.title}</h2>\n{html}"
        )

print(f"Found {len(table_sections)} sections containing tables")

# Combine all table sections into one context
context = "\n\n".join(table_sections)

# Ask questions

questions = [
    "What was Google's operating margin for Q1 2024?",
    "What percentage of revenues is net income?",
    "How much did operating income exceed costs and expenses?",
    "Which expense category contributes the most to total costs?",
]

for question in questions:
    resp = llm.complete(
        f"""
You are given financial tables extracted from a PDF.
Read the tables carefully and answer the question.
If a calculation is needed, explain it briefly.

Question: {question}

Tables:
{context}
"""
    )
    print(f"Q: {question}")
    print(f"A: {resp.text}")
    print("-" * 80)