import os
from sentence_transformers import SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import VectorDBQA
from PyPDF2 import PdfReader

# Function to read content from a file (PDF or DOCX)
def read_file_content(file_path):
    """
    Reads the content of a PDF or DOCX file.
    
    Args:
        file_path (str): Path to the file.
        
    Returns:
        str: Extracted text from the file.
    """
    if file_path.lower().endswith(".pdf"):
        try:
            reader = PdfReader(file_path)
            content = ""
            for page in reader.pages:
                content += page.extract_text() or ""
            return content.strip()
        except Exception as e:
            print(f"Error reading PDF file {file_path}: {e}")
            return ""
    else:
        print(f"Unsupported file format: {file_path}")
        return ""

# Function to compare file content with FAISS vector store using LangChain
def compare_file_with_faiss(file_path, faiss_index_path, model_name="thenlper/gte-large"):
    """
    Compares the content of a file with the vectors in a FAISS index using LangChain.
    
    Args:
        file_path (str): Path to the file to compare.
        faiss_index_path (str): Path to the FAISS index file.
        model_name (str): The name of the embedding model. Defaults to "thenlper/gte-large".
        
    Returns:
        list: A list of similarity scores between the file content and the FAISS index vectors.
    """
    # Read content from the file
    file_content = read_file_content(file_path)
    
    if not file_content:
        print("File content is empty or could not be read.")
        return []

    # Load the FAISS index
    vector_store = FAISS.load_local(faiss_index_path, SentenceTransformerEmbeddings(SentenceTransformer(model_name)))

    # Generate embedding for the file content
    embedding = SentenceTransformer(model_name).encode(file_content)

    # Perform similarity search in FAISS index using LangChain
    results = vector_store.similarity_search_with_score(file_content, k=5)  # Top 5 nearest neighbors

    # Return the similarity scores and the corresponding document texts (optional)
    return results

# Example usage
if __name__ == "__main__":
    # File path to compare
    file_path = "/content/drive/MyDrive/demoAIthinking/giáo trình chính trị/Chủ nghĩa xã hội khoa học/[TailieuVNU.com] Giáo trình Chủ Nghĩa Xã Hội Khoa Học CNXHKH (Không chuyên).pdf"

    # Path to your FAISS index file
    faiss_index_path = "/content/drive/MyDrive/demoAIthinking/vectorstore"

    # Compare file content with FAISS index
    similarity_scores = compare_file_with_faiss(file_path, faiss_index_path)

    print("Similarity Results:")
    for score, text in similarity_scores:
        print(f"Score: {score:.4f}")
        print(f"Text: {text[:200]}...")  # Print the first 200 characters of the matched text
