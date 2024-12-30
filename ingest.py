import argparse
from PyPDF2 import PdfReader
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS


# Custom Embeddings Class
class STEmbeddings(Embeddings):
    def __init__(self, model_name="thenlper/gte-large"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode(text, show_progress_bar=False)
        return embedding.tolist()


# Text Splitting Function
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


# Load PDF Function
def load_pdfs_with_metadata(pdfs, doc_type="generic"):
    text_chunks = []
    metadata = []

    for pdf in pdfs:
        try:
            reader = PdfReader(pdf)
            for page_num, page in enumerate(reader.pages, start=1):
                raw_text = page.extract_text()
                if not raw_text.strip():
                    continue

                chunks = split_paragraphs(raw_text)
                if chunks:
                    text_chunks.extend(chunks)
                    metadata.extend([{"doc_type": doc_type, "page_number": page_num}] * len(chunks))
        except Exception as e:
            print(f"Error reading {pdf}: {e}")

    return text_chunks, metadata


# Load DOCX Function
def load_docx_with_metadata(docx_files, doc_type="generic"):
    text_chunks = []
    metadata = []

    for docx_file in docx_files:
        try:
            doc = Document(docx_file)
            raw_text = "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())

            if not raw_text.strip():
                continue

            chunks = split_paragraphs(raw_text)
            if chunks:
                text_chunks.extend(chunks)
                metadata.extend([{"doc_type": doc_type}] * len(chunks))
        except Exception as e:
            print(f"Error reading {docx_file}: {e}")

    return text_chunks, metadata


# Main Function
def main():
    parser = argparse.ArgumentParser(description="Process PDF and DOCX files into a vector store.")
    parser.add_argument("--pdfs", nargs="*", default=[], help="List of PDF files to process.")
    parser.add_argument("--docx", nargs="*", default=[], help="List of DOCX files to process.")
    parser.add_argument("--doc_type", type=str, default="generic", help="Type of document (e.g., textbook, research paper).")
    parser.add_argument("--output_path", type=str, default="./vectorstore", help="Path to save the vector store.")
    parser.add_argument("--model_name", type=str, default="thenlper/gte-large", help="Model name for embeddings.")
    args = parser.parse_args()

    embeddings = STEmbeddings(model_name=args.model_name)
    text_chunks = []
    metadata = []

    # Load PDFs
    if args.pdfs:
        pdf_chunks, pdf_metadata = load_pdfs_with_metadata(args.pdfs, doc_type=args.doc_type)
        text_chunks.extend(pdf_chunks)
        metadata.extend(pdf_metadata)

    # Load DOCX
    if args.docx:
        docx_chunks, docx_metadata = load_docx_with_metadata(args.docx, doc_type=args.doc_type)
        text_chunks.extend(docx_chunks)
        metadata.extend(docx_metadata)

    if not text_chunks:
        print("No valid text found in the provided files.")
        return

    # Create and save the FAISS vector store
    store = FAISS.from_texts(text_chunks, embeddings, metadatas=metadata)
    store.save_local(args.output_path)
    print(f"Vector store created and saved successfully at {args.output_path}.")


if __name__ == "__main__":
    main()
