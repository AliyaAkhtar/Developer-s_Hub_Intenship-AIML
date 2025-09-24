import os
import fitz  
import weaviate
import openai
from fastapi import FastAPI, UploadFile, File
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import Filter
from dotenv import load_dotenv
from fastapi import APIRouter
import re
import numpy as np

from datetime import datetime
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Weaviate
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_weaviate.vectorstores import WeaviateVectorStore

import weaviate.classes.config as wvcc

# from langchain.embeddings import SentenceTransformerEmbeddings

from weaviate.classes.query import Filter
import uuid
from langchain.prompts import PromptTemplate

# from langchain_community.chat_models import ChatGroq
from langchain_groq import ChatGroq
from langchain_community.embeddings import SentenceTransformerEmbeddings


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

rag_router = APIRouter()



load_dotenv()

# Dictionary to keep memory per pdf_id (or could use user_id + pdf_id)
chat_memories = {}

# Load environment variables
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# print("url: ", WEAVIATE_URL)
# print("api key: ", WEAVIATE_API_KEY)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
headers = {
    "X-OpenAI-Api-Key": OPENAI_API_KEY,
}

if not WEAVIATE_URL or not WEAVIATE_API_KEY:
    raise ValueError("Weaviate credentials are not set.")

# Connect to Weaviate
def get_weaviate_client():
    """Ensure the Weaviate client is always connected."""
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
            headers=headers,
            skip_init_checks=True,
        )
        return client
    except Exception as e:
        raise ValueError(f"Error connecting to Weaviate: {e}")

client = get_weaviate_client()

def clean_filename(filename):
    """Remove special characters from filename to use as collection name."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', os.path.splitext(filename)[0])

# Extract text from PDF
def extract_text_from_pdf(pdf_path, chunk_size=500):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def generate_embedding(text):
    """Generate embeddings using sentence-transformers"""
    embedding = embedding_model.encode(text, normalize_embeddings=True)  
    return embedding.tolist()  

# Upload PDF and store embeddings in Weaviate
@rag_router.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = None
    try:
        file_path = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        chunks = extract_text_from_pdf(file_path)
        client = get_weaviate_client()

        pdf_id = str(uuid.uuid4())  
        collection_name = "pdf_documents"

        existing_collections = client.collections.list_all()

       #Step 1: Check if the collection already exists
        if not client.collections.exists(collection_name):
            client.collections.create(
                name=collection_name,
                description="Collection for storing PDFs",
                # vectorizer_config=weaviate.classes.config.Configure.Vectorizer.text2vec_openai(),
                vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
                properties=[
                    Property(name="pdf_id", data_type=DataType.TEXT, description="Unique ID for the PDF"),
                    Property(name="content", data_type=DataType.TEXT, description="The main content of the document"),
                    Property(name="metadata", data_type=DataType.TEXT, description="Metadata about the document"),
                ],
            )

        #Step 2: Insert the new PDF into the collection
        collection = client.collections.get(collection_name)

        #Insert new data into the existing collection
        for chunk in chunks:
            collection.data.insert({
                "pdf_id": pdf_id,
                "content": chunk,
                "metadata": file.filename,
            })

        return {"pdf_id": pdf_id, "message": f"PDF '{file.filename}' uploaded successfully"}

    except Exception as e:
        return {"error": str(e)}

    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)


# Retrieval-Augmented Generation (RAG) Chat API
@rag_router.post("/chat/")
async def chat_with_pdf(pdf_id: str, query: str):
    try:
        client = get_weaviate_client()

        print("client: ", client)

        collection_name = "Pdf_documents"  

        # Ensure collection exists
        existing_collections = client.collections.list_all()
        
        print("existing collections: ",client.collections.list_all())

        if collection_name not in existing_collections:
            return {"error": f"No documents found."}

        # Initialize the LangChain Weaviate retriever
        retriever = WeaviateVectorStore(
            client=client,
            index_name=collection_name,
            text_key="content",
            # embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
            embedding=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
        ).as_retriever(search_kwargs={"filters": Filter.by_property("pdf_id").equal(pdf_id), "k": 50})

        retrieved_docs = retriever.get_relevant_documents(query)

        print("retrieved docss: ", retrieved_docs)

        if not retrieved_docs:
            return {"error": "No relevant documents found. Try rephrasing your query."}

        retrieved_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Define a strict prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="You are an assistant answering questions using the provided document excerpts. "
                "Base your answer strictly on the given excerpts, but summarize if needed. "
                "If the answer isn't clear, try to infer meaning while staying true to the excerpts.\n\n"
                "Excerpts:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

        # Initialize memory (one per PDF session)
        if pdf_id not in chat_memories:
            chat_memories[pdf_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        memory = chat_memories[pdf_id]

        # llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model="deepseek-r1-distill-llama-70b",
            temperature=0,
            max_tokens=None,
            reasoning_format="parsed",
            timeout=None,
            max_retries=2,
        )

        # rag_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type="stuff", chain_type_kwargs={"prompt": prompt_template})

        # Conversational RAG
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate(
                    input_variables=["context", "question", "chat_history"],
                    template=(
                        "You are a context-aware assistant answering based on documents and chat history.\n\n"
                        "Chat History:\n{chat_history}\n\n"
                        "Excerpts:\n{context}\n\n"
                        "Question: {question}\n\n"
                        "Answer:"
                    ),
                )
            },
        )

        # response = rag_chain.run(query)
        response = qa_chain({"question": query})

        print("Response: ", response)

        # return {"answer": response}
        return {
            "answer": response["answer"],
            "chat_history": [
                {"role": m.type, "content": m.content} for m in memory.chat_memory.messages
            ],
        }
    
    except Exception as e:
        return {"error": str(e)}


@rag_router.get("/get_documents/{pdf_id}")
def get_documents(pdf_id: str):
    client = get_weaviate_client()
    collection_name = "Pdf_documents"  

    existing_collections = client.collections.list_all()
    if collection_name not in existing_collections:
        return {"error": f"Collection '{collection_name}' not found."}

    try:
        collection = client.collections.get(collection_name)

        results = collection.query.fetch_objects(
            filters=Filter.by_property("pdf_id").equal(pdf_id)
        )

        documents = [{"uuid": item.uuid, "content": item.properties["content"]} for item in results.objects]

        if not documents:
            return {"message": "No documents found for this pdf_id."}

        return {"pdf_id": pdf_id, "documents": documents}

    except Exception as e:
        return {"error": f"Error retrieving documents: {e}"}

@rag_router.post("/mock_chat/")
async def mock_chat_with_pdf(pdf_filename: str, query: str):
    return {
        "answer": "This is a mocked response for testing purposes.",
        "timestamp": datetime.utcnow().isoformat()
    }
