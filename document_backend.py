import json
import pandas as pd
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.docstore.document import Document
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings
)
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv

load_dotenv()
gk=os.environ['GROQ_API_KEY']
client=Groq(api_key=gk)

model_name="openai/gpt-oss-20b"
embedding_model=SentenceTransformerEmbeddings(model_name='thenlper/gte-large')

import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename="interaction_logs.jsonl",  # JSON Lines format for easy parsing
    level=logging.INFO,
    format="%(message)s"
)

def log_interaction(query, retrieved_chunks, llm_response, metadata=None):
    """Logs interaction data in JSON format for audit."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "response": llm_response,
        "metadata": metadata or {}
    }
    logging.info(json.dumps(log_entry))


def create_vector_db(file_paths):
    """Load multiple files (PDF/TXT), create vector DB, and set retriever."""
    all_docs = []

    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(path)
            docs = loader.load()
        elif ext == ".txt":
            loader = TextLoader(path)
            docs = loader.load()
        else:
            continue

        all_docs.extend(docs)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_documents(all_docs)

    # Create vectorstore in memory
    vectorstore = Chroma.from_documents(chunks, embedding_model)
    return vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 5})

def generate_answer(prompt, retriever):
    """Generate answer using the retriever from uploaded document."""

    qna_system_message = """
    You are a study assistant that enables conversational interaction, helping users
    ask questions, summarize chapters, quiz themselves, and retain knowledge better.
    Use only the provided context to answer questions.
    """

    # Retrieve relevant docs
    relevant_chunks = retriever.get_relevant_documents(prompt)
    context_list = [d.page_content for d in relevant_chunks]
    context_for_query = ". ".join(context_list)

    qna_user_message_template = """
    ###Context
    {context}

    ###Question
    {question}
    """

    prompt = [
        {'role': 'system', 'content': qna_system_message},
        {'role': 'user', 'content': qna_user_message_template.format(
            context=context_for_query,
            question=prompt
        )}
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=prompt,
            temperature=0
        )
        # Log interaction
        log_interaction(
            query=prompt,
            retrieved_chunks=context_list,
            llm_response=response.choices[0].message.content.strip(),
            metadata={
                "num_chunks": len(context_list),
                "retriever_k": 5,
                "model": model_name
            }
        )
        return response.choices[0].message.content.strip(), context_for_query
    except Exception as e:
        return f"❌ Error: {e}"