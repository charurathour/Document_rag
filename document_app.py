import streamlit as st
from document_backend import generate_answer, create_vector_db
import os

# Page settings
st.set_page_config(page_title="Basic Chatbot", layout="wide")

st.title("Document Chatbot")

with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload your document", type=["pdf", "txt"])
    
    if uploaded_file:

        if "retriever" not in st.session_state:
            with st.spinner("Processing document..."):
                # Save uploaded file temporarily
                temp_path = os.path.join("temp_upload", uploaded_file.name)
                os.makedirs("temp_upload", exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                # Create vector DB
                st.session_state.retriever = create_vector_db(temp_path)
                st.success("✅ Document processed and vector DB created!")

# Store messages
if "messages" not in st.session_state:
   st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

# Show chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])


if user_input := st.chat_input("Type your message here..."):

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)
    # Temporary bot reply (for testing only)
    with st.spinner("Thinking..."):
        answer = generate_answer(user_input, st.session_state.get("retriever", None))
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").markdown(answer)
