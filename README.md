# Political Education Chatbot - UIT

## Overview
An AI-powered chatbot designed to answer questions about political and philosophical subjects in Vietnamese education. It is built with a FastAPI backend and a vanilla JavaScript frontend.

## Features

### Frontend
- Real-time chat interface for seamless interaction.
- PDF file upload functionality with progress tracking.
- Visual feedback for various processing states.
- Responsive design ensuring compatibility across devices.

### Backend
- Efficient PDF processing and knowledge extraction.
- BM25 ranking algorithm for determining context relevance.
- FAISS vector store for similarity-based searches.
- Integration with Google Gemini 1.5 Pro for advanced AI responses.

## Tech Stack
- **Backend**: FastAPI, LangChain, FAISS, Sentence Transformers
- **Frontend**: HTML5, CSS3, JavaScript
- **AI**: Google Generative AI (Gemini 1.5 Pro)

## Installation

### Backend Setup
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the FastAPI server:
   ```bash
   uvicorn app:app --reload --port 8000
   ```

3. (Optional) Serve the frontend:
   ```bash
   python -m http.server 8080
   ```
   Alternatively, open `index.html` in a browser.

## Usage

1. Start the backend server.
2. Open the frontend interface in your browser.
3. (Optional) Upload PDF files to enrich the chatbot's knowledge base.
4. Type questions into the chat input field.
5. Receive detailed, AI-generated answers.

## API Endpoints

### Upload PDF
- **Endpoint**: `POST /upload`
- **Description**: Accepts PDF files for processing and adds their content to the knowledge base.
- **Response**: Returns the processing status.
- **Note**: Includes a 30-second processing delay to ensure thorough extraction.

### Ask Question
- **Endpoint**: `POST /answer`
- **Request Body**:
  ```json
  {
      "query": "Your question",
      "use_external_knowledge": true
  }
  ```
- **Response**: Provides the AI-generated answer along with the inference time.

## UI Components
- **Chat Header**: Displays the chatbot's title and logo for easy identification.
- **Message History Display**: Shows a chronological history of the conversation.
- **File Upload Section**: Enables users to upload PDFs with progress indicators for real-time updates.
- **Question Input**: Includes a text box and send button for submitting queries.

## Environment Variables
- `GOOGLE_API_KEY`: Your Google API key for integrating Gemini 1.5 Pro.

## Development
This chatbot was developed at the University of Information Technology - VNUHCM to enhance learning and engagement in political and philosophical education.

