# README.md

## 📖 Project Overview

This repository implements a Retrieval-Augmented Generation (RAG) chatbot powered by a Qdrant vector database and an NVIDIA-hosted LLM (Meta LLaMA 3.3‑70B). The pipeline covers document ingestion, chunking, embedding, vector store creation, and a Streamlit front‑end for interactive question answering.

### 🔍 Architecture & Flow

```
Raw documents (PDF, TXT)
  └─> preprocess.py (load + split into chunks)
         └─> 2,000‑char chunks w/200‑char overlap
               └─> HuggingFaceEmbeddings (all‑MiniLM‑L6‑v2)
                     └─> Qdrant vector store (collection: amlgo‑docs)
                           └─> rag_pipeline.py (RetrievalQA chain)
                                 └─> ChatNVIDIA LLM (llama‑3.3‑70b‑instruct)
                                       └─> app.py (Streamlit UI)
```

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone <repo-url>
cd Code
pip install -r requirements
```

### 2. Preprocessing & Embeddings

1. Place your PDF/TXT files in `../data/`
2. Run chunking and ingestion:
   ```bash
   python preprocess.py
   ```
   - Splits documents into 2,000‑char chunks
   - Generates embeddings via `sentence‑transformers/all‑MiniLM‑L6‑v2`
   - Creates or overwrites Qdrant collection `amlgo‑docs`

### 3. Build RAG Pipeline

No separate step required — the first call to the Streamlit app will initialize the Retriever + LLM chain.

### 4. Run the Chatbot (Streaming)

```bash
streamlit run app.py
```

- Opens UI at `http://localhost:8501`
- Enter your NVIDIA API key as env var or modify `rag_pipeline.py`
- Queries stream in real‑time; history panel shows previous Q&A

## 🤖 Models & Embeddings

- **Embedding Model**: `sentence‑transformers/all‑MiniLM‑L6‑v2` (384‑dim cosine)
- **LLM**: `meta/llama‑3.3‑70b‑instruct` via `ChatNVIDIA` (streaming enabled)
- **Vector DB**: Qdrant (self‑hosted or cloud, COSINE distance)

## 🎯 Sample Queries & Output

1. **Query:** "What are our terms regarding ebay"
   **Answer:** Our terms regarding eBay are outlined in the User Agreement, which includes provisions such as complying with all terms of the agreement, not breaching or circumventing laws or regulations, not failing to pay for items purchased or deliver items sold, not manipulating prices or interfering with other users' listings, and not taking actions that may undermine the feedback or ratings systems.

   The full context used to generate this answer is: The provided text, which includes the eBay User Agreement, outlines the terms and conditions for using eBay's services. The agreement includes provisions such as complying with all terms of the agreement, not breaching or circumventing laws or regulations, not failing to pay for items purchased or deliver items sold, not manipulating prices or interfering with other users' listings, and not taking actions that may undermine the feedback or ratings systems. Additionally, the agreement covers payments, including the use of designated eBay entities to facilitate payments, and the requirements for buyers and sellers to accept and comply with the Payments Terms of Use. The agreement also includes an Agreement to Arbitrate, which requires users to submit claims to binding and final arbitration and provides for amendments to the agreement, with 30 days' notice to users.

2. **Query:** "15 + 55"
   **Answer:** The answer to the question is 70.

   The full context used to generate the answer is not applicable in this case, as the question "15 + 55" is a simple arithmetic operation that does not require any context from the provided text to be solved. The provided text appears to be a legal agreement or terms of service for eBay, and does not contain any relevant information for solving the mathematical question.

3. **Query:** "what are the policies if i am a resident of California"
   **Answer:** If you are a California resident, you may report complaints to the Complaint Assistance Unit of the Division of Consumer Services of the California Department of Consumer Affairs by contacting them in writing at 400 R Street, Sacramento, CA 95814, or by telephone at (800) 952-5210, in accordance with Cal. Civ. Code §1789.3. 
   
   The full context used to generate the answer is: The provided text mentions that if you are a California resident, in accordance with Cal. Civ. Code §1789.3, you may report complaints to the Complaint Assistance Unit of the Division of Consumer Services of the California Department of Consumer Affairs by contacting them in writing at 400 R Street, Sacramento, CA 95814, or by telephone at (800)952-5210. This information is specifically related to California residents and provides a clear procedure for reporting complaints.

