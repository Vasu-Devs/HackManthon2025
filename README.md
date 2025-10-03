# College Assistant - AI-Powered RAG System

A comprehensive college assistant application built with React frontend, Node.js authentication service, and FastAPI AI backend with RAG (Retrieval-Augmented Generation) capabilities.

## 🎥 Live Demo

https://github.com/Vasu-Devs/HackManthon2025/assets/demo.mp4

*📹 Full system demonstration showing AI chat, document management, and admin features*

## � Demo Videolege Assistant - AI-Powered RAG System

A comprehensive college assistant application built with React frontend, Node.js authentication service, and FastAPI AI backend with RAG (Retrieval-Augmented Generation) capabilities.

## � Demo Video

Watch our comprehensive demo showcasing the College Assistant's features:

[![College Assistant Demo](https://img.shields.io/badge/▶️_Watch_Demo-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://github.com/Vasu-Devs/HackManthon2025/blob/main/Screen%20Recording%202025-09-27%20063342.mp4)

### 🎬 What's Covered in the Demo:
- **🔐 Authentication Flow** - Student and admin login systems with role-based access
- **💬 AI Chat Interface** - Real-time streaming responses with document retrieval context
- **📄 Document Management** - Upload, process, and approve PDF documents with vector embeddings
- **📊 Admin Dashboard** - Upload analytics, document management, and system monitoring
- **� Analytics Dashboard** - Visual charts and statistics using Recharts
- **� Verification Interface** - Document testing and validation system

## �🏗️ Project Architecture

```
e:\GFG\
├── frontend/          # React 19 + Vite 6 frontend (Port 5173)
├── Auth/             # Express + MongoDB authentication service (Port 4000)
└── backend-ai/       # FastAPI + AI/RAG backend (Port 8000)
```

## 🚀 Current Features

### 🤖 AI Chat Assistant
- **Streaming chat responses** via `/chat_stream` endpoint with Google Gemini 2.0 Flash
- **RAG (Retrieval-Augmented Generation)** with real-time document context display
- **Voice chat support** via `/voice_chat` endpoint with microphone integration
- **Department-specific responses** with user context
- **Real-time thinking process** showing document retrieval and processing steps

### 📄 Document Management System
- **Async PDF upload** via `/upload_pdf_async` with real-time progress tracking
- **Document approval workflow** via `/approve_doc/{filename}` (processing → verified)
- **Vector database storage** using ChromaDB + Ollama nomic-embed-text embeddings
- **Document CRUD operations** (upload, download via `/get_file/{filename}`, delete)
- **Status monitoring** with processing states and chunk information

### 👥 Authentication & Authorization
- **JWT-based authentication** with Express + MongoDB
- **Role-based access control** (user/admin) with protected routes
- **bcryptjs password hashing** for secure storage
- **CORS-enabled** frontend-backend communication
- **Login/logout** with localStorage token management

### 📊 Admin Dashboard Features
- **Real-time upload analytics** with progress monitoring
- **System activity logs** via `/logs` endpoint (refreshes every 5s)
- **Document approval interface** with status management
- **File drag-and-drop upload** with validation

### 🎯 Analytics Dashboard
- **Visual charts** using Recharts (Pie, Line, Bar charts)
- **Real-time data visualization** with responsive design
- **Navigation integration** between dashboard components

### 🔍 Verification Interface
- **Document testing and validation** system
- **Admin-only access** for system verification

## 🛠️ Tech Stack

### Frontend (React 19 + Vite 6)
- **React 19.1.1** with modern hooks and JSX transforms
- **Vite 6** for fast development and HMR
- **Tailwind CSS 4.1.13** for utility-first styling
- **React Router DOM 7.9.2** for client-side routing
- **Lucide React 0.544.0** for modern SVG icons
- **React Markdown 10.1.0** with GitHub Flavored Markdown support
- **Framer Motion 12.23.21** for smooth animations
- **Recharts 3.2.1** for data visualization
- **Axios 1.12.2** for HTTP requests
- **JWT Decode 4.0.0** for token handling

### Authentication Service (Express + MongoDB)
- **Express 5.1.0** web framework
- **MongoDB + Mongoose 8.18.2** for user data storage
- **JWT (jsonwebtoken 9.0.2)** for token-based authentication
- **bcryptjs 3.0.2** for password hashing
- **Winston 3.17.0** for structured logging
- **CORS 2.8.5** for cross-origin requests

### AI Backend (FastAPI + LangChain)
- **FastAPI** with uvicorn for async Python API
- **Google Generative AI SDK** for Gemini 2.0 Flash responses
- **ChromaDB** for persistent vector database storage
- **Ollama** with nomic-embed-text model for embeddings
- **LangChain Community** for document processing and retrieval
- **pdfplumber + PyPDF2** for PDF text extraction
- **Python Multipart** for file upload handling
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

### Authentication Endpoints (Port 4000)
- `POST /auth/register` - Register new user with role assignment
- `POST /auth/login` - User login with JWT token generation
- Protected routes via JWT middleware

### FastAPI Endpoints (Port 8000)
- `GET /health` - Backend health check with vectorstore status
- `POST /chat` - Regular chat with AI (non-streaming)
- `POST /chat_stream` - **Streaming chat responses** with real-time tokens
- `POST /admin_chat` - Admin-specific chat interface
- `POST /voice_chat` - Voice-to-voice interaction
- `POST /upload_pdf_async` - **Async PDF upload** with progress tracking
- `GET /upload_status/{upload_id}` - Check upload progress status
- `POST /approve_doc/{filename}` - Approve document for vectorstore
- `GET /documents` - List all uploaded documents with metadata
- `POST /test_retrieval` - Test document retrieval functionality
- `GET /recent_activities` - Get recent system activities
- `GET /logs` - System logs for monitoring (refreshed every 5s)
- `GET /get_file/{filename}` - Download uploaded documents

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

### Frontend Structure & Routing
```
frontend/src/
├── components/          # React components
│   ├── App.jsx                 # Main router with 6 routes
│   ├── LandingPage.jsx         # "/" - Landing page
│   ├── DashboardLayout.jsx     # "/dashboard" - Admin dashboard
│   ├── ChatInterface.jsx       # "/chat" - AI chat interface
│   ├── Analytics.jsx           # "/analytics" - Analytics dashboard
│   ├── DocumentsList.jsx       # "/policies" - Document management
│   ├── VerificationInterface.jsx # "/verification" - Testing interface
│   ├── Hero.jsx                # Landing page hero section
│   ├── FeatureCards.jsx        # Feature showcase cards
│   ├── Dashboard.jsx           # Dashboard subcomponent
│   ├── Signup.jsx              # Auth modal component
│   ├── StudentBlackPanel.jsx   # Student auth panel
│   ├── AdminBlackPanel.jsx     # Admin auth panel
│   ├── StudentWhitePanel.jsx   # Student login form
│   ├── AdminWhitePanel.jsx     # Admin login form
│   ├── LanguageSwitcher.jsx    # Language selector
│   └── Footer.jsx              # Footer component
├── services/           # API service layer (created but commented out)
│   └── api.js         # FastAPI integration utilities
├── config/            # Configuration files  
│   └── api.js        # API endpoints configuration
├── assets/            # Static assets (React logo)
└── icons/            # SVG icons for UI
```

### Backend Structure
```
backend-ai/
├── app.py                    # FastAPI main application with 12 endpoints
├── requirements.txt          # Python dependencies (26 packages)
├── .env                     # Environment variables configuration
├── chroma_db/               # ChromaDB vector database persistence
├── temp_md/                 # Temporary markdown processing directory
├── uploaded_docs/           # Permanent PDF document storage
├── chat_logs.csv           # Chat interaction logging
└── venv/                   # Python virtual environment

Auth/
├── index.js                 # Express server with MongoDB connection
├── package.json            # Node.js dependencies (8 packages)
├── .env                    # MongoDB URI and JWT secret
├── models/                 # Mongoose user models
├── routes/                 # Authentication and API routes
├── middleware/             # JWT and logging middleware
├── logs/                   # Winston structured logs
└── utils/                  # Utility functions
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

## 📋 Current Implementation Status

### ✅ Fully Implemented Features
- **Frontend Routes**: 6 main routes with React Router DOM
- **Authentication System**: Complete JWT-based auth with MongoDB
- **AI Chat Interface**: Streaming responses with real-time document context
- **Document Management**: Upload, process, approve, delete workflows
- **Admin Dashboard**: File upload with drag-and-drop, progress tracking
- **Analytics Dashboard**: Charts and visualizations with Recharts
- **Vector Database**: ChromaDB with Ollama embeddings for RAG
- **API Integration**: Direct fetch calls to FastAPI (service layer exists but not active)

### 🔧 Service Layer Status
- **Created but Inactive**: `src/services/api.js` and `src/config/api.js` exist
- **Current Approach**: Components use direct `fetch()` calls to `http://127.0.0.1:8000/*`
- **Reason**: Service layer imports caused build issues, temporarily disabled
- **Future**: Service layer ready for re-integration when build issues resolved

### 🎯 Key Working Features
- **Real-time Streaming**: `/chat_stream` endpoint with token-by-token responses
- **Document Processing**: Async upload with chunking progress display  
- **Role-based Access**: Admin vs User routing and permissions
- **Voice Integration**: `/voice_chat` endpoint for audio interactions
- **System Monitoring**: Live activity logs refreshing every 5 seconds
- **Responsive Design**: Works across desktop, tablet, and mobile devices

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
