import json
import pandas as pd
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
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

qna_system_message = """
You are an assistant to a firm who answers user queries.
User input will have the context required by you to answer user questions.
This context will begin with the token: ###Context.
The context contains references to specific portions of a document relevant to the user query.

User questions will begin with the token: ###Question.

Please answer user questions using only the context provided in the input.
"""

qna_user_message_template = """
###Context
Here are some documents that are relevant to the question mentioned below.
{context}

###Question
{question}
"""
def create_vector_db(pdf_path):
    """Load PDF, create vector DB, and set retriever."""
    
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Create vectorstore in memory (no need to persist)
    vectorstore = Chroma.from_documents(chunks, embedding_model)
    return vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 1})

def generate_answer(prompt, retriever):
    """Generate answer using the retriever from uploaded document."""

    qna_system_message = """
    You are an assistant to a firm who answers user queries.
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
        return response.choices[0].message.content.strip(), context_for_query
    except Exception as e:
        return f"❌ Error: {e}"