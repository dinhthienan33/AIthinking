import os
import time
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Initialize embeddings
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
    """
    Process uploaded PDFs and split their text into chunks.
    """
    text_chunks = []
    metadata = []
    lock = ThreadPoolExecutor()

    def process_pdf(uploaded_file):
        try:
            reader = PdfReader(uploaded_file)
            for page_num, page in enumerate(reader.pages, start=1):
                raw_text = page.extract_text()
                if not raw_text.strip():
                    continue

                chunks = split_paragraphs(raw_text)
                with lock:
                    text_chunks.extend(chunks)
                    metadata.extend([{"doc_type": doc_type, "page_number": page_num}] * len(chunks))
        except Exception as e:
            logger.error(f"Error reading file: {e}")

    with ThreadPoolExecutor() as executor:
        executor.map(process_pdf, uploaded_files)

    return text_chunks, metadata

def retrieval_result(query, path, top_k=3):
    """
    Retrieve relevant documents from the FAISS vector store.
    """
    try:
        store = FAISS.load_local(path, embeddings=embeddings, allow_dangerous_deserialization=True)
        retriever = store.as_retriever()
        retrieved_docs = retriever.get_relevant_documents(query)[:top_k]
        if not retrieved_docs:
            return "No relevant information found.", ""
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return context
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        return "", ""

def answer_query(query, context):
    """
    Answer a query using a language model with the given context.
    """
    try:
        api_key = os.environ("GOOGLE_API_KEY")  # Replace "default-key" with actual default or raise error
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            google_api_key=api_key
        )

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

        chain = LLMChain(llm=llm, prompt=prompt_template)
        response = chain.invoke({"context": context, "query": query})
        return response["text"]
    except Exception as e:
        logger.error(f"Error during query answering: {e}")
        return "Error generating answer."

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QuestionRequest(BaseModel):
    query: str
    use_external_knowledge: bool = False

class AnswerResponse(BaseModel):
    answer: str
    inference_time: float

# API Endpoints
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        os.makedirs('./external_knowledge', exist_ok=True)
        contents = await file.read()
        pdf_file = io.BytesIO(contents)
        text_chunks, metadata = load_pdfs_with_metadata([pdf_file])
        store = FAISS.from_texts(text_chunks, embeddings, metadatas=metadata)
        store.save_local('./external_knowledge')
        pdf_file.close()
        return {"message": "File uploaded and processed successfully"}
    except Exception as e:
        logger.error(f"Error during file upload: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/answer", response_model=AnswerResponse)
async def get_answer(request: QuestionRequest):
    start_time = time.time()
    try:
        context = retrieval_result(request.query, './vectorstore')
        if request.use_external_knowledge:
            external_context = retrieval_result(request.query, './external_knowledge')
            context = context + "\n\n" + external_context

        answer = answer_query(request.query, context)
        inference_time = time.time() - start_time
        return AnswerResponse(answer=answer, inference_time=inference_time)
    except Exception as e:
        logger.error(f"Error during query answering: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
