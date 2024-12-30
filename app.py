import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer   
import os
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.document import Document
# Define custom embedding class
class STEmbeddings(Embeddings):
    def __init__(self, model_name="thenlper/gte-large"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode(text)
        return embedding.tolist()

embeddings = STEmbeddings()
def split_paragraphs(raw_text):
    """
    Splits text into semantically meaningful chunks using RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?"],
        length_function=len
    )
    return text_splitter.split_text(raw_text)
def load_pdfs_with_metadata(uploaded_files, doc_type="generic"):
    text_chunks = []
    metadata = []

    for uploaded_file in uploaded_files:
        try:
            reader = PdfReader(uploaded_file)
            for page_num, page in enumerate(reader.pages, start=1):
                raw_text = page.extract_text()
                if not raw_text.strip():
                    continue

                chunks = split_paragraphs(raw_text)
                if chunks:
                    text_chunks.extend(chunks)
                    metadata.extend([{"doc_type": doc_type, "page_number": page_num}] * len(chunks))
        except Exception as e:
            print(f"Error reading {uploaded_file.name}: {e}")

    return text_chunks, metadata
def retrieval_result(text,path,top_k=5):
    store = FAISS.load_local(path, embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = store.as_retriever()
    retrieved_docs = retriever.get_relevant_documents(text)[:top_k]
    if not retrieved_docs:
        return "No relevant information found.", ""
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context
def answer_query(query,context):
    """Answers a query using the indexed vector store."""
    # Load FAISS vector store
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        google_api_key="AIzaSyDU5w4BHDCzovzRG2cyEbqXdYXyfnmIstU",  # Replace with your actual API key
    )

    # Define a prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template=(
            "Bạn là một giảng viên môn kinh tế chính trị tại Việt Nam. Dựa vào thông tin sau để trả lời câu hỏi của sinh viên:\n\n"
            "Chỉ cần đưa ra nguyên văn câu trả lời đúng, không trả lời gì thêm.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{query}\n\n"
            "Answer:"
        )
    )

    # Create the LLM chain
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Generate the answer
    response = chain.run({"context": context, "query": query})
    return response

def main():
    st.title("Hệ thống hỏi đáp về các môn chính trị tại UIT")
    st.write("Nếu bạn muốn sử dụng thêm kiến thức, hãy upload file vào đây ")

    # File upload section
    uploaded_file = st.file_uploader("Upload File (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=False)
    st.write("Note: Each file must be less than 5MB.")
    if st.button("Upload file"):
        if uploaded_file:
            # use load_pdfs_with_metadata 
            text_chunks, metadata = load_pdfs_with_metadata([uploaded_file])
            # save the text_chunks and metadata to a file
            store = FAISS.from_texts(text_chunks, embeddings, metadatas=metadata)
            store.save_local('./external_knowledge')       
            st.success("Upload successful")
        else:
            st.error("No file uploaded.")
    # Query section
    query = st.text_area("Hãy nhập câu hỏi dạng single choice:", height=340)
    submit = st.button("Submit")

    if query and submit:
        try:
            if uploaded_file:
                external_context = retrieval_result(query, './external_knowledge')
            else:
                external_context = ""
            context = retrieval_result(query, './vectorstore')
            context = context + "\n\n" + external_context
            answer = answer_query(query, context)
            st.subheader("Câu trả lời :")
            st.write(answer)
            # st.subheader("Retrieved Context:")
            # st.write(context)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Hãy nhập câu hỏi.")

if __name__ == "__main__":
    main()