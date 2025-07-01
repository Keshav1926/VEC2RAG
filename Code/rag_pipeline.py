from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from qdrant_client import QdrantClient
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

    return qdrant.as_retriever(search_kwargs={'k': 5})

def build_llm(streaming: bool = True):
    return ChatNVIDIA(
        model="meta/llama-3.1-405b-instruct",
        api_key=NVIDIA_API_KEY,
        temperature=0.0,
        max_tokens=1024,
        top_p=0.7,
        streaming=streaming
    )

def format_docs(docs):
    """Format retrieved documents for the prompt"""
    return "\n\n".join(doc.page_content for doc in docs)

def build_streaming_chain():
    """Build a streaming-capable chain using LCEL (LangChain Expression Language)"""
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

    def get_context(inputs):
        docs = retriever.invoke(inputs["query"])
        return {"context": format_docs(docs), "question": inputs["query"]}

    streaming_chain = (
        RunnablePassthrough.assign(context_and_question=get_context)
        | {
            "context": lambda x: x["context_and_question"]["context"],
            "question": lambda x: x["context_and_question"]["question"]
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return streaming_chain
