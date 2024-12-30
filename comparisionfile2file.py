from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
from docx import Document
import os

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
    elif file_path.lower().endswith(".docx"):
        try:
            doc = Document(file_path)
            return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
        except Exception as e:
            print(f"Error reading DOCX file {file_path}: {e}")
            return ""
    else:
        print(f"Unsupported file format: {file_path}")
        return ""

def calculate_file_similarity(file1, file2, model_name="thenlper/gte-large"):
    """
    Calculates the similarity score between two document files (PDF or DOCX) using embeddings.

    Args:
        file1 (str): Path to the first file.
        file2 (str): Path to the second file.
        model_name (str): The name of the embedding model. Defaults to "thenlper/gte-large".

    Returns:
        float: Cosine similarity score between the two documents (range: 0 to 1).
    """
    # Read content from the files
    doc1 = read_file_content(file1)
    doc2 = read_file_content(file2)

    if not doc1 or not doc2:
        print("One or both files could not be read or are empty.")
        return 0.0

    # Load the SentenceTransformer model
    model = SentenceTransformer(model_name)

    # Generate embeddings for both documents
    embedding1 = model.encode(doc1, convert_to_tensor=True)
    embedding2 = model.encode(doc2, convert_to_tensor=True)

    # Calculate cosine similarity
    similarity_score = util.cos_sim(embedding1, embedding2).item()

    return similarity_score
if __name__ == "__main__":
    # Paths to your files
    file1_path = "path/to/your/document1.pdf"
    file2_path = "path/to/your/document2.docx"

    # Calculate similarity
    similarity = calculate_file_similarity(file1_path, file2_path)
    print(f"Similarity Score: {similarity:.4f}")
