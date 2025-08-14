# Document_rag

  ## Video Link:
https://drive.google.com/file/d/1Jvhv8De5I1O-2DbPDXEjzEzLKu0eL60a/view?usp=drive_link

## Document Uploader & RAG Chatbot

## 🚀 Features
- **Document Upload**: Supports PDF, TXT (extendable to other formats).
- **Chunking & Embeddings**: Splits documents into chunks and generates vector embeddings.
- **RAG Chatbot**: Answers questions based on uploaded documents using an LLM + Vector Database.
- **Interactive UI**: Built with Streamlit for easy document uploads and chatting.

---

## 🛠️ Tech Stack
- **Frontend/UI**: Streamlit
- **Backend**: Python
- **Vector Store**: Chroma
- **Embeddings Model**: HuggingFace
- **LLM**: OpenAI GPT

  ## 📦 Installation Steps

### Clone the Repository
```bash
git clone https://github.com/charurathour/Document_rag.git
cd Document_rag

### Create and Activate Virtual Environment
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

### Install Dependencies
pip install -r requirements.txt

### Set Environment Variables
Create a .env file in the root directory:
OPENAI_API_KEY=your_openai_api_key_here

### Run the Application
streamlit run document_app.py


