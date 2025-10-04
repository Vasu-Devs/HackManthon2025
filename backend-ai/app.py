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
from fastapi import Request

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import httpx

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://localhost:4000")


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
UPLOAD_DIR = Path("./uploaded_docs")  # Permanent storage for uploaded PDFs

os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(TEMP_MD_DIR, exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

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

# ----------------EMAIL--------------------------
async def get_user_email(user_id: str, token: str) -> str:
    """
    Fetch user email from Express auth service using regNo + JWT token.
    """
    try:
        async with httpx.AsyncClient() as client:
            # Call your /auth/me to verify & get user payload
            resp = await client.get(f"{AUTH_SERVICE_URL}/auth/me", headers={"Authorization": f"Bearer {token}"})
            if resp.status_code != 200:
                return None
            payload = resp.json()
            regNo = payload.get("regNo")

            # Now call a custom route (you can add /api/user/email/:regNo in Express)
            user_resp = await client.get(f"{AUTH_SERVICE_URL}/api/user/{regNo}", headers={"Authorization": f"Bearer {token}"})
            if user_resp.status_code == 200:
                return user_resp.json().get("email")
    except Exception as e:
        print("❌ get_user_email failed:", e)
    return None


def send_email(to_email: str, subject: str, body: str):
    """
    Send an email via SendGrid and raise on failure.
    Requires:
      SENDGRID_API_KEY
      SENDGRID_SENDER  -> must be a verified Single Sender OR a domain-authenticated address
    """
    sg_api_key = os.getenv("SENDGRID_API_KEY")
    sender = os.getenv("SENDGRID_SENDER")

    if not sg_api_key or not sender:
        raise Exception("Missing SENDGRID_API_KEY or SENDGRID_SENDER in environment")

    message = Mail(
        from_email=sender,         # must match a verified sender in SendGrid
        to_emails=to_email,        # can be any recipient
        subject=subject,
        plain_text_content=body
    )

    sg = SendGridAPIClient(sg_api_key)
    response = sg.send(message)

    # SendGrid returns 202 on success (accepted)
    if response.status_code != 202:
        raise Exception(f"SendGrid error status={response.status_code}, body={response.body}")
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
    model = genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config={
            "temperature": 0,
            "max_output_tokens": 1800,
            "top_p": 0.8,
            "candidate_count": 1
        }
    )
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
    print("📧 SENDGRID_API_KEY loaded:", bool(os.getenv("SENDGRID_API_KEY")))
    print("📧 SENDGRID_SENDER:", os.getenv("SENDGRID_SENDER"))

    print("🔥 Gemini backend ready!")

# -------------------- CHAT HELPERS --------------------
GREETING_PATTERNS = re.compile(r"^\s*(hi|hello|hey+|yo|hola|namaste|bonjour|👋)\s*!*$", re.IGNORECASE)
CLOSING_PATTERNS = re.compile(r"^\s*(thanks|thank you|ok|bye|goodbye|good night|cool|great)\s*!*$", re.IGNORECASE)

def is_greeting(msg: str) -> bool:
    return bool(GREETING_PATTERNS.match(msg.strip()))

def is_closing(msg: str) -> bool:
    return bool(CLOSING_PATTERNS.match(msg.strip()))

# -------------------- ENDPOINTS --------------------
@app.get("/health")
def health():
    return {"status": "ok", "vectorstore_loaded": _vectorstore is not None}


@app.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(
    payload: ChatRequest = Body(...), 
    user_id: str = "default", 
    background_tasks: BackgroundTasks = None,
    request: Request = None   # <-- inject Request here
):
    if _vectorstore is None:
        return ChatResponse(
            response="Vectorstore not initialized. Upload PDFs first.",
            department=payload.department,
            sources=[],
            elapsed_seconds=0.0,
        )

    if user_id not in _chat_history:
        _chat_history[user_id] = []
    
    msg = payload.message.strip()

    # --- Greeting / Closing shortcuts ---
    if is_greeting(msg):
        return ChatResponse(
            response="Hey there! How can I help you today?",
            department=payload.department, 
            sources=[], 
            elapsed_seconds=0.0
        )

    if is_closing(msg):
        return ChatResponse(
            response="You're welcome! Have a great day 🚀",
            department=payload.department, 
            sources=[], 
            elapsed_seconds=0.0
        )

    # --- Detect reminder intent with Gemini ---
    intent_prompt = f"""
    Determine if this student query is asking to set a reminder.
    Query: "{msg}"

    Respond ONLY with one word: "reminder" or "normal".
    """
    intent = (await run_gemini(intent_prompt)).strip().lower()

    if "reminder" in intent:
        # ✅ Proper JWT extraction from request headers
        auth_header = request.headers.get("authorization") if request else None
        token = auth_header.split(" ")[1] if auth_header else None

        if not token:
            return ChatResponse(
                response="⚠️ Missing token, cannot verify your account for reminder.",
                department=payload.department, 
                sources=[], 
                elapsed_seconds=0.0
            )

        to_email = await get_user_email(user_id, token)
        if not to_email:
            return ChatResponse(
                response="⚠️ Could not find your email in the system to set a reminder.",
                department=payload.department, 
                sources=[], 
                elapsed_seconds=0.0
            )

        # --- NEW: Ask Gemini to draft the reminder email ---
        reminder_prompt = f"""
        You are an assistant that creates reminder emails.

        The student with regNo = {user_id} asked: "{msg}"

        Task:
        - Detect what they want to be reminded about.
        - Write a short, polite email as the College Assistant.
        - Tone: professional but friendly.
        - Include emojis if suitable.
        - Respond only with the email body text, no explanations.
        """

        email_body = await run_gemini(reminder_prompt)

        subject = "📌 College Assistant Reminder"
        body = f"""
Hello {user_id},

{email_body.strip()}

Best regards,  
College Assistant Bot
        """

        # Send email asynchronously
        if background_tasks:
            background_tasks.add_task(send_email, to_email, subject, body)
        else:
            send_email(to_email, subject, body)

        log_activity(f"📧 Reminder email sent to {to_email}")

        return ChatResponse(
            response=f"✅ Reminder has been set! An email was sent to {to_email}.",
            department=payload.department, 
            sources=[], 
            elapsed_seconds=0.0
        )

    # --- Normal RAG flow ---
    _chat_history[user_id].append({"role": "student", "content": payload.message})
    conversation = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in _chat_history[user_id]])

    retriever = _vectorstore.as_retriever(search_kwargs={"k": payload.k or DEFAULT_K})
    docs = retriever.get_relevant_documents(payload.message)
    approved = [d for d in docs if _doc_approval_status.get(d.metadata.get("source"), False)]

    if not approved:
        return ChatResponse(
            response="I'm not fully certain about this. Let me connect you with the department staff for confirmation.",
            department=payload.department,
            sources=[],
            elapsed_seconds=0.0
        )

    context_snippets = "\n\n".join([d.page_content[:500] for d in approved])
    
    prompt = f"""
You are **College Assistant**, an intelligent multilingual virtual agent for the {payload.department} department. 
Your role is to resolve student queries directly, clearly, and confidently.

🔥 IMPORTANT RULE:
🚫 Do NOT explain your reasoning, steps, or options. 
🚫 Do NOT output fallback apologies in multiple languages. 
✅ Only give the final assistant reply, short and confident.

🎯 TASK
Student has asked: "{payload.message}"

Reply in the same language as the student, with a short, friendly assistant message.
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
async def chat_stream(payload: ChatRequest = Body(...), request: Request = None):
    query = payload.message.strip()

    # --- Detect reminder intent with Gemini ---
    intent_prompt = f"""
    Determine if this student query is asking to set a reminder.
    Query: "{query}"

    Respond ONLY with one word: "reminder" or "normal".
    """
    intent = (await run_gemini(intent_prompt)).strip().lower()

    # --- Handle reminder intent ---
    if "reminder" in intent:
        # ✅ Extract token from headers
        auth_header = request.headers.get("authorization") if request else None
        token = auth_header.split(" ")[1] if auth_header else None

        if not token:
            def event_stream():
                yield json.dumps({"type": "token", "text": "⚠️ Missing token, cannot set reminder."}) + "\n"
                yield json.dumps({"type": "done"}) + "\n"
            return StreamingResponse(event_stream(), media_type="application/json")

        # Lookup user email
        to_email = await get_user_email("default", token)
        if not to_email:
            def event_stream():
                yield json.dumps({"type": "token", "text": "⚠️ Could not find your email in the system."}) + "\n"
                yield json.dumps({"type": "done"}) + "\n"
            return StreamingResponse(event_stream(), media_type="application/json")

        # Ask Gemini to draft email
        reminder_prompt = f"""
        You are an assistant that creates reminder emails.

        The student asked: "{query}"

        Task:
        - Detect what they want to be reminded about.
        - Write a short, polite email as the College Assistant.
        - Tone: professional but friendly.
        - Include emojis if suitable.
        - Respond only with the email body text, no explanations.
        """
        email_body = await run_gemini(reminder_prompt)

        subject = "📌 College Assistant Reminder"
        body = f"""
Hello,

{email_body.strip()}

Best regards,  
College Assistant Bot
        """

        try:
            send_email(to_email, subject, body)
            log_activity(f"📧 Reminder email sent to {to_email}")
            def event_stream():
                yield json.dumps({"type": "status", "message": "📧 Reminder email sent"}) + "\n"
                yield json.dumps({"type": "token", "text": f"✅ Reminder set! Email sent to {to_email}."}) + "\n"
                yield json.dumps({"type": "done"}) + "\n"
            return StreamingResponse(event_stream(), media_type="application/json")
        except Exception as e:
            err = str(e)
            print("❌ send_email failed:", err)
            def event_stream():
                yield json.dumps({"type": "status", "message": "❌ Email failed"}) + "\n"
                yield json.dumps({"type": "token", "text": f"Couldn’t send the reminder email: {err}"}) + "\n"
                yield json.dumps({"type": "done"}) + "\n"
            return StreamingResponse(event_stream(), media_type="application/json")


    # --- Normal streaming RAG flow ---
    context_prompt = f"""
You are a multilingual college assistant for the {payload.department} department. 
Student query: "{query}"

🔥 IMPORTANT RULE:
🚫 Do NOT explain your reasoning, steps, or options. 
🚫 Do NOT output fallback apologies in multiple languages. 
✅ Only give the final assistant reply, short and confident.

Only return the assistant’s reply text.
Do not include reasoning, analysis, or options.
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

@app.post("/set_reminder")
async def set_reminder(
    user_id: str = "default",
    background_tasks: BackgroundTasks = None,
    token: str = Body(..., embed=True)  # frontend must send {"token": "..."}
):
    """
    Manually set a reminder for a student.
    Looks up user email from Express service and sends email.
    """

    # Lookup user email from Express Auth service
    to_email = await get_user_email(user_id, token)
    if not to_email:
        raise HTTPException(status_code=404, detail="User email not found")

    subject = "📌 College Assistant Reminder"
    body = f"""
Hello {user_id},

This is your reminder from the College Assistant.
You asked me to remind you about something — consider this your nudge 😊.

Best regards,
College Assistant Bot
    """

    if background_tasks:
        background_tasks.add_task(send_email, to_email, subject, body)
    else:
        send_email(to_email, subject, body)

    log_activity(f"📧 Manual reminder email sent to {to_email}")
    return {"message": f"✅ Reminder set and email sent to {to_email}"}

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

    # Permanent copy (Option 2)
    try:
        (UPLOAD_DIR / file.filename).write_bytes(content)
    except Exception as e:
        upload_progress[upload_id] = {"status": "error", "error": f"Failed to save file: {e}"}
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    background_tasks.add_task(process_pdf_background, upload_id, content, file.filename)

    log_activity(f"📤 Upload started: {file.filename}")
    return {"upload_id": upload_id, "message": "Upload started"}

async def process_pdf_background(upload_id: str, file_content: bytes, filename: str):
    try:
        temp_path = Path(TEMP_MD_DIR) / filename
        temp_path.write_bytes(file_content)

        # Initial status with known filename
        upload_progress[upload_id] = {
            "status": "processing",
            "progress": 0,
            "processed_chunks": 0,
            "total_chunks": 0,
            "filename": filename
        }

        # Step 1: Extract text
        text = extract_text_from_pdf(temp_path)
        text = preprocess_text(text)

        # Step 2: Chunk text
        chunks = chunk_text(text)
        if not chunks:
            upload_progress[upload_id] = {
                "status": "error",
                "error": "No valid text",
                "filename": filename,
                "processed_chunks": 0,
                "total_chunks": 0
            }
            return

        total = len(chunks)

        # ✅ Update total_chunks immediately so frontend sees it
        upload_progress[upload_id]["total_chunks"] = total

        metadata = [{"source": filename, "chunk_id": i} for i in range(total)]

        # Step 3: Embed per chunk for accurate progress updates
        global _vectorstore
        if _vectorstore is None:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            _vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

        for i, chunk in enumerate(chunks):
            _vectorstore.add_texts(texts=[chunk], metadatas=[metadata[i]])

            processed = i + 1
            pct = int((processed / total) * 100)

            # ✅ Update progress after EACH chunk
            upload_progress[upload_id] = {
                "status": "processing",
                "progress": pct,
                "processed_chunks": processed,
                "total_chunks": total,
                "filename": filename
            }

            # Yield control so frontend poller can catch the update
            await asyncio.sleep(0.2)

        # Step 4: Mark doc as uploaded but not approved
        _doc_approval_status[filename] = False

        upload_progress[upload_id] = {
            "status": "completed",
            "progress": 100,
            "num_chunks": total,
            "processed_chunks": total,
            "total_chunks": total,
            "filename": filename
        }

        log_activity(f"✅ Upload completed: {filename}")

        try:
            temp_path.unlink()
        except:
            pass

    except Exception as e:
        upload_progress[upload_id] = {
            "status": "error",
            "error": str(e),
            "filename": filename,
            "processed_chunks": 0,
            "total_chunks": 0
        }

@app.get("/upload_status/{upload_id}")
def get_upload_status(upload_id: str):
    return upload_progress.get(upload_id, {
        "status": "not_found",
        "progress": 0,
        "processed_chunks": 0,
        "total_chunks": 0,
    })

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
        # 1. Remove from vectorstore
        if _vectorstore is not None:
            _vectorstore._collection.delete(where={"source": filename})

        # 2. Remove approval status
        _doc_approval_status.pop(filename, None)

        # 3. Remove physical file from UPLOAD_DIR
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            file_path.unlink()

        log_activity(f"🗑️ Deleted: {filename}")
        return {"message": f"{filename} deleted completely (vectorstore + storage)"}

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
    """
    Lists documents based on permanent storage in UPLOAD_DIR so they survive restarts.
    Also shows approval status from in-memory map (defaults to 'processing' if not set).
    """
    docs = []
    try:
        for path in sorted(UPLOAD_DIR.glob("*")):
            if not path.is_file():
                continue
            filename = path.name
            approved = _doc_approval_status.get(filename, False)
            try:
                size = path.stat().st_size
                uploaded_ts = path.stat().st_mtime
                uploaded_iso = datetime.fromtimestamp(uploaded_ts).isoformat()
                docs.append({
                    "name": filename,
                    "status": "verified" if approved else "processing",
                    "size": size,
                    "date": uploaded_iso.split("T")[0],
                    "time": uploaded_iso.split("T")[1][:8],
                })
            except Exception:
                docs.append({
                    "name": filename,
                    "status": "verified" if approved else "processing",
                    "size": None,
                    "date": None,
                    "time": None,
                })
        return docs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")

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

@app.get("/get_file/{filename}")
def get_file(filename: str):
    """
    Streams a permanently stored PDF from UPLOAD_DIR.
    """
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path), filename=filename, media_type="application/pdf")

# -------------------- RUN --------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)  