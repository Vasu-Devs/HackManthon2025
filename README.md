# HackManthon2025

A comprehensive AI-powered application with FastAPI backend and modern React frontend.

## 📋 Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)  
- [Setup Instructions](#setup-instructions)
- [Backend AI Setup](#backend-ai-setup)
- [Frontend Setup](#frontend-setup)
- [Usage](#usage)
- [Contributing](#contributing)

## 🏗️ Project Structure

```
HackManthon2025/
├── backend-ai/          # AI backend service with FastAPI
│   ├── app.py          # Main application file
│   ├── requirements.txt # Python dependencies
│   └── .env            # Environment variables
├── frontend/           # React frontend application
│   ├── src/           # Source code
│   ├── package.json   # Node.js dependencies
│   └── vite.config.js # Vite configuration
├── data/              # Data files and documents
└── README.md          # This file
```

## 🔧 Prerequisites

Before setting up this project, ensure you have the following installed:

- **Python 3.12+** (for backend-ai)
- **Node.js 16+** and **npm** (for frontend)
- **Git** (for version control)
- **CUDA-capable GPU** (optional but recommended for faster embeddings)

### Installing PyTorch with CUDA Support

For optimal performance with AI embeddings and GPU acceleration:

1. **Check CUDA Version**: Run `nvidia-smi` to check your CUDA version
2. **Install PyTorch with CUDA 12.1**: The setup instructions below include the correct PyTorch installation
3. **Verify GPU**: The application will automatically detect and use GPU if available

## 🚀 Setup Instructions

### 1. Fork and Clone the Repository

```bash
# Fork the repository on GitHub first, then clone your fork
git clone https://github.com/YOUR_USERNAME/HackManthon2025.git
cd HackManthon2025
```

### 2. Backend AI Setup

#### Step 1: Navigate to Backend Directory
```bash
cd backend-ai
```

#### Step 2: Create Virtual Environment with Python 3.12
```bash
# If you have multiple Python versions installed
py -3.12 -m venv venv
# Or use the default Python
python -m venv venv
```

#### Step 3: Activate Virtual Environment

**On Windows:**
```bash
./venv/Scripts/Activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

#### Step 4: Install Dependencies with GPU Support
```bash
# First, install PyTorch with CUDA 12.1 support
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Then install all other dependencies
pip install -r requirements.txt
```

#### Step 5: Download Embedding Model
The application uses **BAAI/bge-m3** embedding model from HuggingFace which will download automatically (~2GB) on first run.

#### Step 6: Environment Configuration
1. Create a `.env` file in the `backend-ai` directory
2. Add your Google API key:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

#### Step 7: Run the Backend Server
```bash
# Make sure you're in the backend-ai directory and venv is activated
uvicorn app:app --reload
```

**Note**: On first run, the HuggingFace BGE-M3 embedding model will be automatically downloaded (~2GB). This may take several minutes depending on your internet connection.

The backend server should now be running on `http://localhost:8000`

### 3. Frontend Setup

#### Step 1: Navigate to Frontend Directory
```bash
# From the root directory
cd frontend
```

#### Step 2: Install Node.js Dependencies
```bash
npm install
```

#### Step 3: Start Development Server
```bash
npm run dev
```

The frontend application should now be running on `http://localhost:5173`

## 🎯 Usage

1. **Start the Backend**: Ensure the backend-ai server is running (Step 2.7 above)
2. **Start the Frontend**: Ensure the frontend development server is running (Step 3.3 above)  
3. **Access the Application**: Open your browser and navigate to `http://localhost:5173`

## 📝 API Documentation

Once the backend is running, you can access the interactive API documentation at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🧠 AI Models & Technology

### Embedding Model: BAAI/BGE-M3
- **Model**: `BAAI/bge-m3` from HuggingFace
- **Type**: Multilingual embedding model with excellent semantic understanding
- **Size**: ~2GB (downloads automatically on first run)
- **Features**: 
  - Supports multiple languages
  - High-quality semantic embeddings for RAG
  - GPU acceleration with CUDA 12.1 support
  - Automatic fallback to CPU if GPU unavailable

### Language Model: Google Gemini 2.0 Flash
- **Model**: `gemini-2.0-flash`
- **Provider**: Google AI
- **Features**: Fast, efficient responses with excellent reasoning capabilities

## 🔧 Development

### Backend Development
- The backend uses **FastAPI** with **LangChain** for AI operations
- **ChromaDB** for vector database functionality
- **HuggingFace BGE-M3** embeddings for semantic search and RAG
- **Google Gemini 2.0 Flash** for AI chat responses
- **PyTorch with CUDA 12.1** for efficient GPU-accelerated inference

### Frontend Development
- Built with **React** and **Vite**
- Modern JavaScript/TypeScript development
- Hot module replacement for fast development

## 🛠️ Troubleshooting

### Common Issues

1. **Virtual Environment Activation Issues**
   - On Windows, if `./venv/Scripts/Activate` doesn't work, try: `venv\Scripts\activate.bat`
   - Ensure you're in the `backend-ai` directory
   - Make sure you're using Python 3.12+ with `py -3.12 -m venv venv`

2. **PyTorch/CUDA Installation Issues**
   - Verify CUDA version with `nvidia-smi`
   - Reinstall PyTorch with correct CUDA version:
     ```bash
     pip uninstall torch -y
     pip install torch --index-url https://download.pytorch.org/whl/cu121
     ```
   - Check GPU detection in the application logs

3. **Embedding Model Download Issues**
   - Ensure you have a stable internet connection for the initial model download
   - The BGE-M3 model (~2GB) downloads automatically from HuggingFace
   - If download fails, delete the model cache and restart: check `~/.cache/huggingface/` directory
   - For GPU issues, ensure PyTorch is properly installed with CUDA support

4. **Port Conflicts**
   - Backend default port: 8000
   - Frontend default port: 5173
   - Change ports in configuration files if needed

5. **Dependencies Installation Issues**
   - For Python: Make sure virtual environment is activated and using Python 3.12+
   - For Node.js: Delete `node_modules` and `package-lock.json`, then run `npm install` again

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is part of HackManthon2025. Please check with the organizers for licensing information.

## 🆘 Support

If you encounter any issues during setup:
1. Check the troubleshooting section above
2. Ensure all prerequisites are properly installed
3. Verify that all steps were followed in order
4. Make sure you're using Python 3.12+ and PyTorch with CUDA 12.1
5. Create an issue on GitHub with detailed error messages

---

**Happy Hacking! 🚀**
