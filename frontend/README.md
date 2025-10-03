# College Assistant - AI-Powered RAG System

A comprehensive college assistant application built with React frontend, Node.js authentication service, and FastAPI AI backend with RAG (Retrieval-Augmented Generation) capabilities.

## 🏗️ Project Architecture

```
e:\GFG\
├── frontend/          # React + Vite frontend
├── Auth/             # Node.js + MongoDB authentication service  
└── backend-ai/       # FastAPI + AI/RAG backend
```

## 🚀 Features

### 🤖 AI Chat Assistant
- **Streaming chat responses** powered by Google Gemini 2.0 Flash
- **RAG (Retrieval-Augmented Generation)** with document context
- **Real-time document retrieval** display during conversations
- **Department-specific responses** (General, CS, etc.)
- **Voice chat support** with microphone integration

### 📄 Document Management
- **PDF upload and processing** with progress tracking
- **Document approval workflow** (processing → verified)
- **Vector database storage** using ChromaDB + Ollama embeddings
- **Admin document management** (approve, delete, view status)
- **Real-time document status updates**

### 👥 User Authentication
- **JWT-based authentication** with role management (user/admin)
- **MongoDB user storage** with bcrypt password hashing
- **Role-based routing** and access control
- **Registration and login system**

### 📊 Admin Dashboard
- **Upload analytics** and progress monitoring
- **Recent activity logs** and system monitoring
- **Document approval interface**
- **System health monitoring**

## 🛠️ Tech Stack

### Frontend (React + Vite)
- **React 19** with modern hooks
- **Vite** for fast development
- **Tailwind CSS** for styling
- **React Router** for navigation
- **Lucide React** for icons
- **React Markdown** with GitHub Flavored Markdown
- **Framer Motion** for animations
- **Recharts** for analytics

### Authentication Service (Node.js)
- **Express.js** web framework
- **MongoDB** with Mongoose ODM
- **JWT** for token-based authentication
- **bcrypt** for password hashing
- **Winston** for logging

### AI Backend (FastAPI)
- **FastAPI** for high-performance API
- **Google Gemini 2.0 Flash** for LLM responses
- **ChromaDB** for vector database
- **Ollama** with nomic-embed-text for embeddings
- **LangChain** for document processing
- **PyPDF2 & pdfplumber** for PDF text extraction
- **Python 3.8+** runtime

## 📋 Prerequisites

Before running the application, ensure you have:

- **Node.js** (v16 or higher)
- **Python** (3.8 or higher)
- **MongoDB** (local or cloud instance)
- **Ollama** installed with `nomic-embed-text` model
- **Google API Key** for Gemini 2.0 Flash

### Install Ollama and Model
```bash
# Install Ollama (Windows/Mac/Linux)
# Visit: https://ollama.ai/download

# Pull the required embedding model
ollama pull nomic-embed-text
```

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd GFG
```

### 2. Authentication Service Setup
```bash
cd Auth
npm install

# Create .env file
echo "MONGODB_URI=mongodb://localhost:27017/college_assistant
JWT_SECRET=your_super_secret_jwt_key_here
PORT=4000" > .env

# Start the auth service
npm start
```

### 3. FastAPI Backend Setup
```bash
cd backend-ai

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GOOGLE_API_KEY=your_google_gemini_api_key
PERSIST_DIR=./chroma_db
DEFAULT_K=5
CHUNK_SIZE=800
CHUNK_OVERLAP=100" > .env

# Start FastAPI server
python app.py
```

### 4. Frontend Setup
```bash
cd frontend
npm install

# Start development server
npm run dev
```

## 🌐 Application URLs

- **Frontend**: http://localhost:5173
- **Auth Service**: http://localhost:4000
- **FastAPI Backend**: http://localhost:8000
- **FastAPI Docs**: http://localhost:8000/docs

## 📚 API Documentation

### Authentication Endpoints
- `POST /auth/register` - Register new user
- `POST /auth/login` - User login
- `GET /auth/health` - Service health check

### FastAPI Endpoints
- `GET /health` - Backend health check
- `POST /chat` - Regular chat with AI
- `POST /chat_stream` - Streaming chat responses
- `POST /upload_pdf_async` - Upload PDF documents
- `GET /documents` - List all documents
- `POST /approve_doc/{filename}` - Approve document (admin)
- `DELETE /delete_doc/{filename}` - Delete document (admin)
- `GET /recent_activities` - Get recent system activities

## 🔧 Development

### Frontend Development
```bash
cd frontend
npm run dev      # Start dev server
npm run build    # Build for production
npm run lint     # Run ESLint
npm run preview  # Preview production build
```

### Backend Development
```bash
cd backend-ai
python app.py   # Start with auto-reload
```

### Auth Service Development
```bash
cd Auth
npm run dev     # Start with nodemon (if configured)
npm start       # Start production server
```

## 📁 Project Structure

### Frontend Structure
```
frontend/src/
├── components/          # React components
│   ├── ChatInterface.jsx       # Main chat interface
│   ├── DocumentsList.jsx       # Document management
│   ├── DashboardLayout.jsx     # Admin dashboard
│   ├── LandingPage.jsx         # Landing page
│   └── ...
├── services/           # API service layer
│   └── api.js         # FastAPI integration
├── config/            # Configuration files
│   └── api.js        # API endpoints config
└── assets/           # Static assets
```

### Backend Structure
```
backend-ai/
├── app.py                    # FastAPI main application
├── requirements.txt          # Python dependencies
├── chroma_db/               # Vector database storage
├── temp_md/                 # Temporary file processing
└── uploaded_docs/           # Document storage
```

## 🔐 Environment Variables

### Auth Service (.env)
```env
MONGODB_URI=mongodb://localhost:27017/college_assistant
JWT_SECRET=your_jwt_secret_key
PORT=4000
```

### FastAPI Backend (.env)
```env
GOOGLE_API_KEY=your_google_gemini_api_key
PERSIST_DIR=./chroma_db
DEFAULT_K=5
CHUNK_SIZE=800
CHUNK_OVERLAP=100
TEMP_MD_DIR=./temp_md
BATCH_SIZE=20
```

## 🧪 Testing

### Test Authentication
```bash
curl -X POST http://localhost:4000/auth/health
```

### Test FastAPI Backend
```bash
curl -X GET http://localhost:8000/health
```

### Test Document Upload
1. Navigate to http://localhost:5173/dashboard
2. Upload a PDF file
3. Monitor processing in real-time

## 📈 Monitoring & Logs

- **Auth Service Logs**: `Auth/logs/app.log`
- **FastAPI Logs**: Console output with structured logging
- **Chat Logs**: `backend-ai/chat_logs.csv` with Gemini-powered analytics
- **Activity Monitoring**: Real-time activities in dashboard

## 🚨 Troubleshooting

### Common Issues

1. **White Screen on Frontend**
   - Check browser console for JavaScript errors
   - Ensure all services are running
   - Verify API endpoints are accessible

2. **FastAPI Import Errors**
   - Ensure virtual environment is activated
   - Install requirements: `pip install -r requirements.txt`

3. **Auth Service Connection Issues**
   - Check MongoDB is running
   - Verify MongoDB URI in .env

4. **Vector Database Issues**
   - Ensure Ollama is running
   - Pull required model: `ollama pull nomic-embed-text`

### Port Conflicts
If ports are occupied, update the following:
- Auth Service: Change `PORT` in `.env`
- FastAPI: Update `port` in `app.py`
- Frontend: Update `vite.config.js`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Google Gemini** for AI responses
- **Ollama** for local embeddings
- **ChromaDB** for vector storage
- **FastAPI** for high-performance backend
- **React** ecosystem for frontend
