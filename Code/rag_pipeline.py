import os
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from langchain_qdrant import Qdrant


QDRANT_URL="https://aa2d8f89-fe71-43d8-9d57-b64a06308cf7.us-west-2-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.arjgJYfVAT2nvEadZEkQBn0dvEhgsbQZntYCrKls8uw"
NVIDIA_API_KEY="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"
COLLECTION_NAME = 'amlgo-docs'
EMBED_MODEL = "all-MiniLM-L6-v2"


def get_retriever():
    # Initialize Qdrant client
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=False,
    )

    # Try different embedding initialization approaches
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={
                'device': 'cpu',
                'trust_remote_code': True
            },
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        print(f"First embedding attempt failed: {e}")
        try:
            # Fallback approach
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
        except Exception as e2:
            print(f"Second embedding attempt failed: {e2}")
            # Final fallback
            print("Using default embedding model without additional kwargs.")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    qdrant = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )

    # Return a retriever interface
    return qdrant.as_retriever(search_kwargs={'k': 5})

def build_llm(streaming: bool = True):
    return ChatNVIDIA(
        model="meta/llama-3.3-70b-instruct",
        api_key=NVIDIA_API_KEY,
        temperature=0.0,
        max_tokens=1024,
        top_p=0.7,
        streaming=streaming
    )

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def build_chain():
    retriever = get_retriever()
    llm = build_llm(streaming=True)

    # Custom prompt
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Answer the following question only based on the context provided and only give the final answer.
        Additionally, write the full context used to generate the answer.

        First paragraph: Provide the answer.
        Second paragraph: write the full context used to generate the answer.

        Context:
        {context}

        Question:
        {question}

        Answer:"""
    )

    # Build QA chain using custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # 'stuff' puts all documents into the prompt
        chain_type_kwargs={"prompt": qa_prompt},
        return_source_documents=True
    )

    return qa_chain


def get_tools():
    qa_chain = build_chain()
    def _qa_fn(query: str) -> str:
        result = qa_chain({'query': query})
        return result['result']
    return [
        Tool(
            name="qa_tool",
            func=_qa_fn,
            description= "Use this tool to answer factual questions based on the documents stored in the knowledge base. "
                         "The tool returns a two-paragraph answer: first with the response, then a markdown list of sources."

        )
    ]