from dotenv import load_dotenv
load_dotenv()
import json
import os
import time
import uuid
import csv
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import asyncio
import warnings
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Body, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# LangChain (for RAG vectorstore)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Google AI SDKs
import google.generativeai as genai

# PDF loaders
import pdfplumber
import PyPDF2
try:
    from langchain_community.document_loaders import PyPDFLoader
    LANGCHAIN_PDF_AVAILABLE = True
except ImportError:
    LANGCHAIN_PDF_AVAILABLE = False

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="langchain")

# -------------------- CONFIG --------------------
PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")
DEFAULT_K = int(os.getenv("DEFAULT_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TEMP_MD_DIR = os.getenv("TEMP_MD_DIR", "./temp_md")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))

os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(TEMP_MD_DIR, exist_ok=True)

executor = ThreadPoolExecutor(max_workers=4)

# Gemini config
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
GEMINI_MODEL = "gemini-2.0-flash"

# CSV log file
LOG_FILE = Path("./chat_logs.csv")

# -------------------- FASTAPI --------------------
app = FastAPI(
    title="College Assistant RAG API (Gemini-powered)",
    version="10.0",
    description="RAG backend with context, Gemini-driven CSV logging, and persistent policies"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- MODELS --------------------
class ChatRequest(BaseModel):
    message: str
    department: Optional[str] = "General"
    k: Optional[int] = DEFAULT_K

class ChatResponse(BaseModel):
    response: str
    department: str
    sources: List[Dict[str, Any]] = []
    elapsed_seconds: float

# -------------------- GLOBALS --------------------
_vectorstore = None
upload_progress = {}
_doc_approval_status: Dict[str, bool] = {}
_chat_history: Dict[str, List[Dict[str, str]]] = {}
_activity_log: List[Dict[str, str]] = []

# -------------------- HELPERS --------------------
def log_activity(message: str):
    _activity_log.insert(0, {
        "message": message,
        "time": datetime.now(timezone.utc).strftime("%H:%M:%S")
    })
    if len(_activity_log) > 20:
        _activity_log.pop()

def extract_text_from_pdf(pdf_path: Path) -> str:
    if pdfplumber:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n\n".join([p.extract_text() or "" for p in pdf.pages])
    if LANGCHAIN_PDF_AVAILABLE:
        loader = PyPDFLoader(str(pdf_path))
        return "\n\n".join([p.page_content for p in loader.load()])
    reader = PyPDF2.PdfReader(open(pdf_path, "rb"))
    return "\n\n".join([p.extract_text() or "" for p in reader.pages])

def preprocess_text(text: str) -> str:
    text = text.replace("&amp;", "&")
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def chunk_text(text: str) -> List[str]:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "],
        length_function=len,
    )
    return [c.strip() for c in splitter.split_text(text) if len(c.strip()) > 50]

def embed_and_store_fast(chunks: List[str], metadata: Optional[List[Dict]] = None):
    global _vectorstore
    if _vectorstore is None:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        _vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    total = len(chunks)
    if total <= BATCH_SIZE:
        _vectorstore.add_texts(texts=chunks, metadatas=metadata)
        return _vectorstore

    futures = []
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        meta_batch = metadata[i:i+BATCH_SIZE] if metadata else None
        futures.append(executor.submit(_vectorstore.add_texts, texts=batch, metadatas=meta_batch))

    for f in futures:
        f.result()
    return _vectorstore

async def run_gemini(prompt: str) -> str:
    model = genai.GenerativeModel(GEMINI_MODEL)
    resp = model.generate_content(prompt)
    return resp.text if resp and resp.text else "No response generated."

async def analyze_with_gemini(query: str, response: str, elapsed: float, docs: List[str]):
    analysis_prompt = f"""
    Analyze this interaction and return JSON only.

    Query: {query}
    Response: {response}
    Retrieved docs: {docs}
    Response time: {elapsed:.3f} seconds

    Required JSON fields:
    {{
      "query_id": "{str(uuid.uuid4())[:8]}",
      "detected_language": "<language of query>",
      "category": "<exam|fees|timetable|hostel|general>",
      "model_confidence": "<0-1 float>",
      "fallback": <true|false>,
      "retrieved_docs": {docs},
      "response_time": {elapsed:.3f},
      "timestamp": "{datetime.now(timezone.utc).isoformat()}"
    }}
    """
    model = genai.GenerativeModel(GEMINI_MODEL)
    resp = model.generate_content(analysis_prompt)
    try:
        return json.loads(resp.text)
    except:
        return None

def append_csv(row: Dict[str, Any]):
    file_exists = LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# -------------------- STARTUP --------------------
@app.on_event("startup")
def startup_event():
    global _vectorstore
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        _vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        print(f"✅ Vector DB loaded from {PERSIST_DIR}")
    except Exception as e:
        print(f"❌ Failed to load vectorstore: {e}")
        _vectorstore = None
    print("🔥 Gemini backend ready!")

# -------------------- ENDPOINTS --------------------
@app.get("/health")
def health():
    return {"status": "ok", "vectorstore_loaded": _vectorstore is not None}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(payload: ChatRequest = Body(...), user_id: str = "default"):
    if _vectorstore is None:
        return ChatResponse(
            response="Vectorstore not initialized. Upload PDFs first.",
            department=payload.department,
            sources=[],
            elapsed_seconds=0.0,
        )

    if user_id not in _chat_history:
        _chat_history[user_id] = []

    _chat_history[user_id].append({"role": "student", "content": payload.message})
    conversation = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in _chat_history[user_id]])

    retriever = _vectorstore.as_retriever(search_kwargs={"k": payload.k or DEFAULT_K})
    docs = retriever.get_relevant_documents(payload.message)
    approved = [d for d in docs if _doc_approval_status.get(d.metadata.get("source"), False)]

    context_snippets = "\n\n".join([d.page_content[:500] for d in approved])

    prompt = f"""
    The following is a conversation between a student and a helpful college assistant specializing in {payload.department}.
    Maintain context across turns. Always assume queries are about THIS college.

    Conversation so far:
    {conversation}

    Relevant documents:
    {context_snippets}

    Assistant:
    """

    start = time.time()
    result_text = await run_gemini(prompt)
    elapsed = time.time() - start

    _chat_history[user_id].append({"role": "assistant", "content": result_text})
    _chat_history[user_id] = _chat_history[user_id][-20:]

    log_activity(f"💬 Chat message: {payload.message[:30]}...")

    # Gemini-driven analysis + CSV logging
    analysis = await analyze_with_gemini(payload.message, result_text, elapsed,
                                         [d.metadata.get("source") for d in approved])
    if analysis:
        append_csv(analysis)

    return ChatResponse(
        response=result_text,
        department=payload.department,
        sources=[{"source": d.metadata.get("source")} for d in approved],
        elapsed_seconds=round(elapsed, 3)
    )

@app.post("/chat_stream")
async def chat_stream(payload: ChatRequest = Body(...)):
    query = payload.message
    context_prompt = f"""
    You are a multilingual college assistant.
    Department: {payload.department}
    Student query: {query}
    """

    def event_stream():
        retriever = _vectorstore.as_retriever(search_kwargs={"k": payload.k or DEFAULT_K})
        docs = retriever.get_relevant_documents(query)
        yield json.dumps({"type": "status", "message": "📚 searching documents"}) + "\n"
        for d in docs:
            preview = d.page_content[:120].replace("\n", " ")
            yield json.dumps({"type": "doc", "source": d.metadata.get("source"), "preview": preview}) + "\n"

        yield json.dumps({"type": "status", "message": "🤔 Generating answer..."}) + "\n"

        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(context_prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield json.dumps({"type": "token", "text": chunk.text}) + "\n"

        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/json")

@app.post("/admin_chat", response_model=ChatResponse)
async def admin_chat(payload: ChatRequest = Body(...)):
    start = time.time()
    prompt = f"You are an admin assistant. Student query: {payload.message}"
    result_text = await run_gemini(prompt)

    retriever = _vectorstore.as_retriever(search_kwargs={"k": payload.k or DEFAULT_K})
    docs = retriever.get_relevant_documents(payload.message)

    elapsed = time.time() - start
    log_activity(f"🛠️ Admin chat: {payload.message[:30]}...")
    return ChatResponse(
        response=result_text,
        department=payload.department,
        sources=[{"source": d.metadata.get("source")} for d in docs],
        elapsed_seconds=round(elapsed, 3)
    )

@app.post("/upload_pdf_async")
async def upload_pdf_async(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    content = await file.read()
    upload_id = f"{file.filename}_{int(time.time())}"
    upload_progress[upload_id] = {"status": "started", "progress": 0}
    background_tasks.add_task(process_pdf_background, upload_id, content, file.filename)

    log_activity(f"📤 Upload started: {file.filename}")
    return {"upload_id": upload_id, "message": "Upload started"}

async def process_pdf_background(upload_id: str, file_content: bytes, filename: str):
    try:
        temp_path = Path(TEMP_MD_DIR) / filename
        temp_path.write_bytes(file_content)
        upload_progress[upload_id] = {"status": "processing", "progress": 0}

        # Step 1: Extract text
        text = extract_text_from_pdf(temp_path)
        text = preprocess_text(text)

        # Step 2: Chunk text
        chunks = chunk_text(text)
        if not chunks:
            upload_progress[upload_id] = {"status": "error", "error": "No valid text"}
            return

        total = len(chunks)
        metadata = [{"source": filename, "chunk_id": i} for i in range(total)]

        # Step 3: Embed in batches with progress updates
        global _vectorstore
        if _vectorstore is None:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            _vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

        for i in range(0, total, BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            meta_batch = metadata[i:i+BATCH_SIZE]
            _vectorstore.add_texts(texts=batch, metadatas=meta_batch)

            # Update progress (% of chunks processed)
            progress = int(((i + len(batch)) / total) * 100)
            upload_progress[upload_id] = {"status": "processing", "progress": progress}

        # Step 4: Mark doc as uploaded but not approved
        _doc_approval_status[filename] = False

        upload_progress[upload_id] = {
            "status": "completed",
            "progress": 100,
            "num_chunks": total
        }

        log_activity(f"✅ Upload completed: {filename}")
        try:
            temp_path.unlink()
        except:
            pass
    except Exception as e:
        upload_progress[upload_id] = {"status": "error", "error": str(e)}


@app.get("/upload_status/{upload_id}")
def get_upload_status(upload_id: str):
    return upload_progress.get(upload_id, {"status": "not_found"})

@app.post("/approve_doc/{filename}")
def approve_doc(filename: str):
    if filename not in _doc_approval_status:
        raise HTTPException(status_code=404, detail="Doc not found")
    _doc_approval_status[filename] = True
    log_activity(f"✔️ Approved: {filename}")
    return {"message": f"{filename} approved and now available to students"}

@app.delete("/delete_doc/{filename}")
def delete_doc(filename: str):
    global _vectorstore
    try:
        _vectorstore._collection.delete(where={"source": filename})
        _doc_approval_status.pop(filename, None)
        log_activity(f"🗑️ Deleted: {filename}")
        return {"message": f"{filename} deleted from vectorstore"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")

@app.post("/test_retrieval")
def test_retrieval(request: ChatRequest):
    if _vectorstore is None:
        return {"status": "error", "message": "No docs uploaded"}
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(request.message)
    return {
        "status": "success",
        "query": request.message,
        "retrieved_docs": [
            {"source": d.metadata.get("source"), "preview": d.page_content[:200]}
            for d in docs
        ]
    }

@app.get("/documents")
def list_documents():
    docs = []
    for filename, approved in _doc_approval_status.items():
        docs.append({
            "name": filename,
            "status": "verified" if approved else "processing"
        })
    return docs

@app.get("/recent_activities")
def get_recent_activities():
    return _activity_log

@app.get("/logs")
def get_logs():
    try:
        if not LOG_FILE.exists():
            return {"error": "log file not found"}
        with LOG_FILE.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        last_logs = [line.strip() for line in lines if line.strip()][-5:]
        return {"logs": last_logs}
    except Exception as e:
        return {"error": str(e)}

# -------------------- RUN --------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
