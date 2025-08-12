import streamlit as st
from document_backend import generate_answer, create_vector_db
import os
import shutil

# Page settings
st.set_page_config(page_title="StudyMate AI", layout="wide")

st.title("StudyMate AI")

with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your documents", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        if "last_uploaded_files" not in st.session_state:
            st.session_state.last_uploaded_files = []
        if uploaded_files != st.session_state.last_uploaded_files:
            with st.spinner("Processing documents..."):
                if os.path.exists("temp_upload"):
                    shutil.rmtree("temp_upload")  # delete folder and all its contents
                os.makedirs("temp_upload")
                saved_paths = []
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join("temp_upload", uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_paths.append(temp_path)

            # Create vector DB for all uploaded docs
            st.session_state.retriever = create_vector_db(saved_paths)
            st.success("✅ All documents processed and vector DB created!")
        st.session_state.last_uploaded_files = uploaded_files

        # 📥 Log file download option
        log_file_path = "interaction_logs.jsonl"
        if os.path.exists(log_file_path):
            with open(log_file_path, "rb") as f:
                st.download_button(
                    label="📥 Download Interaction Logs",
                    data=f,
                    file_name="interaction_logs.jsonl",
                    mime="application/json"
                )


# Store messages
if "messages" not in st.session_state:
   st.session_state.messages = [{"role": "assistant", "content": "👋 Hello! I’m your StudyMate AI.\n\n"
                                "You can:\n"
                                "• ❓ Ask questions about your documents\n"
                                "• 📚 Get summaries of specific chapters or topics\n"
                                "• 🧠 Quiz yourself to test your knowledge\n\n"
                                "What would you like to start with?"}]

# Show chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])


if user_input := st.chat_input("Type your message here..."):

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)
    # Temporary bot reply (for testing only)
    with st.spinner("Thinking..."):
        answer, context = generate_answer(user_input, st.session_state.get("retriever", None))
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").markdown(answer)

    # Show context right after answer
    with st.expander("📄 Retrieved Context"):
        st.markdown(context)