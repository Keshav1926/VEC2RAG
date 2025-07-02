# README.md

## 📖 Project Overview

This repository implements a Retrieval-Augmented Generation (RAG) chatbot powered by a Qdrant vector database and an NVIDIA-hosted LLM (Meta LLaMA-3.1-405B). The pipeline covers document ingestion, chunking, embedding, vector store creation, and a Streamlit front‑end for interactive question answering.

### 🔍 Architecture & Flow

```
Raw documents (PDF, TXT)
  └─> preprocess.py (load + split into chunks)
         └─> 2,000‑char chunks w/200‑char overlap
               └─> HuggingFaceEmbeddings (all‑MiniLM‑L6‑v2)
                     └─> Qdrant vector store (collection: amlgo‑docs)
                           └─> rag_pipeline.py (RetrievalQA chain)
                                 └─> ChatNVIDIA LLM (llama-3.1-405b-instruct)
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
- **LLM**: `llama-3.1-405b-instruct` via `ChatNVIDIA` (streaming enabled)
- **Vector DB**: Qdrant (self‑hosted or cloud, COSINE distance)

## 🎯 Sample Queries & Output

**Query 1**

![Screenshot (43)](https://github.com/user-attachments/assets/94eaf23e-0f0e-4c8d-844b-d57c8e3d428b)


**Query 2**
   
![Screenshot (44)](https://github.com/user-attachments/assets/4135f570-a4d7-40d6-8e41-82940d823a76)


**Query 3**

![Screenshot (45)](https://github.com/user-attachments/assets/80102d2d-f7a2-493b-a114-d2d2caccb1ad)


**Query 4**

![Screenshot (46)](https://github.com/user-attachments/assets/453dd062-e0e9-462c-859d-0670634d56cf)


**Conversation History**

![Screenshot (47)](https://github.com/user-attachments/assets/b6994e84-c06a-4ba5-9483-5c476babc855)

**Clear Buttom**

![Screenshot (48)](https://github.com/user-attachments/assets/aabcc3d4-0bbc-42cf-8c18-f286a702dce1)


**Streaming response Video**



https://github.com/user-attachments/assets/1c64c4ed-36c4-4123-92f7-9f73e76bbd4f


