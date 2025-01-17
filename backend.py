from flask import Flask, request, jsonify
from flask_cors import CORS
from app import answer_query, load_pdfs_with_metadata, embeddings
from langchain.vectorstores import FAISS

app = Flask(__name__)
CORS(app)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query = data['query']
    context = retrieval_result(query, './vectorstore')
    answer = answer_query(query, context)
    return jsonify({'answer': answer})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        text_chunks, metadata = load_pdfs_with_metadata([file])
        store = FAISS.from_texts(text_chunks, embeddings, metadatas=metadata)
        store.save_local('./external_knowledge')
        return jsonify({'message': 'File uploaded successfully'}), 200

if __name__ == '__main__':
    app.run(port=8000)