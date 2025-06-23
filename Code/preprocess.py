import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

BASE_DIR       = os.path.dirname(__file__)
DATA_DIR       = os.path.join(BASE_DIR, '..', 'data')

EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_URL="https://aa2d8f89-fe71-43d8-9d57-b64a06308cf7.us-west-2-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.arjgJYfVAT2nvEadZEkQBn0dvEhgsbQZntYCrKls8uw"



def load_and_split(path: str):
    if path.lower().endswith('.pdf'):
        loader = PyPDFLoader(path)
    else:
        loader = TextLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " "],
    )
    return splitter.split_documents(docs)


def embed_and_store(chunks, collection_name: str):

    # Prepare embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Initialize Qdrant client
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=False,
    )

    # Delete existing collection if present
    try:
        client.delete_collection(collection_name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception as e:
        print(f"Collection {collection_name} doesn't exist or couldn't be deleted: {e}")

    # Determine vector dimension (fallback to 384)
    try:
        # Try to get dimension from the embedding model
        sample_embedding = embeddings.embed_query("test")
        dim = len(sample_embedding)
        print(f"Detected embedding dimension: {dim}")
    except Exception:
        dim = 384
        print(f"Using fallback dimension: {dim}")

    # Create new collection with proper VectorParams
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=dim,
            distance=Distance.COSINE,
        ),
    )
    print(f"Created collection: {collection_name}")

    # Wrap in LangChain Qdrant wrapper and ingest
    qdrant = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    
    print(f"Adding {len(chunks)} chunks to collection...")
    qdrant.add_documents(chunks)
    return qdrant


def main():
    # Load & chunk all documents
    data_files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    if not data_files:
        print(f"No files found in {DATA_DIR}")
        return
    
    paths = [os.path.join(DATA_DIR, f) for f in data_files]
    all_chunks = []
    
    for p in paths:
        if os.path.isfile(p):  # Additional check
            print(f"Processing {p}...")
            chunks = load_and_split(p)
            all_chunks.extend(chunks)
            print(f"  Added {len(chunks)} chunks")

    if not all_chunks:
        print("No chunks were created from the documents.")
        return

    print(f"Total chunks to process: {len(all_chunks)}")
    
    # Embed & store into Qdrant
    embed_and_store(all_chunks, collection_name="amlgo-docs")
    print("✅ Embedding & ingestion complete.")


if __name__ == '__main__':
    main()