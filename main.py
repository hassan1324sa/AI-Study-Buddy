import os
import shutil
import json
import fitz  # PyMuPDF
import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from jose import jwt, JWTError
from passlib.context import CryptContext

from sqlalchemy import create_engine, Column, Integer, String, DateTime, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# ---------------------------
# Load env
# ---------------------------
load_dotenv("app.env")  # file with COHERE_API, JWT_SECRET

COHERE_API = os.getenv("COHERE_API")
if not COHERE_API:
    raise RuntimeError("Please set COHERE_API in app.env")

JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_32_CHARS_MIN")
JWT_ALGO = "HS256"
JWT_EXPIRE_MIN = int(os.getenv("JWT_EXPIRE_MIN", str(60 * 24)))  # default 24h

PERSIST_DIR = os.getenv("CHROMA_DIR", "chroma_db")
os.makedirs(PERSIST_DIR, exist_ok=True)

# ---------------------------
# DB (users) setup (SQLite)
# ---------------------------
DB_URL = os.getenv("USER_DB_URL", "sqlite:///./users.db")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer = HTTPBearer()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(128), nullable=False, unique=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint("username", name="uq_username"),)


Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------
# Auth helpers
# ---------------------------
def create_user(db: Session, username: str, password: str) -> User:
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    user = User(username=username, password_hash=pwd_ctx.hash(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    user = db.query(User).filter(User.username == username).first()
    if not user or not pwd_ctx.verify(password, user.password_hash):
        return None
    return user


def create_token(username: str) -> str:
    exp = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MIN)
    payload = {"sub": username, "exp": exp}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def get_current_user(token: HTTPAuthorizationCredentials = Depends(bearer)) -> str:
    try:
        payload = jwt.decode(token.credentials, JWT_SECRET, algorithms=[JWT_ALGO])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ---------------------------
# LLM + Vectorstore
# ---------------------------
llm = ChatCohere(cohere_api_key=COHERE_API, model="command-r-plus",temperature=0.5)
embeddings = CohereEmbeddings(cohere_api_key=COHERE_API, model="embed-multilingual-v3.0")


def get_user_vectorstore(user_id: str) -> Chroma:
    return Chroma(
        collection_name=f"study_buddy_{user_id}",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


def get_user_qa_chain(user_id: str) -> RetrievalQA:
    vs = get_user_vectorstore(user_id)
    retriever = vs.as_retriever(search_kwargs={"k": 5})
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )


# ---------------------------
# PDF ingestion
# ---------------------------
def add_pdf_to_db(file_bytes: bytes, filename: str, user_id: str):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    vs = get_user_vectorstore(user_id)

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if not text.strip():
            continue
        chunks = splitter.split_text(text)
        metadatas = [{"page": page_num, "source": filename, "user_id": user_id} for _ in chunks]
        vs.add_texts(chunks, metadatas=metadatas)

    vs.persist()


def answer_question(query: str, user_id: str):
    qa = get_user_qa_chain(user_id)
    result = qa.invoke(query)
    return {
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]],
    }


def generate_quiz_from_context(context: str, num_questions: int = 10) -> List[Dict[str, Any]]:
    prompt = f"""
أنت مساعد تعليمي. عندك المحتوى التالي:
{context}

من فضلك أنشئ {num_questions} أسئلة اختيار من متعدد (MCQ).
رجع الناتج كـ JSON list بالشكل التالي:
[
  {{
    "question": "النص",
    "options": ["اختيار1","اختيار2","اختيار3","اختيار4"],
    "answer": "الاختيار الصحيح"
  }},
  ...
]
احرص أن يكون الناتج JSON صالحًا فقط — لا تُرجع نصًا خارج بنية الـ JSON.
"""
    res = llm.invoke(prompt)
    try:
        return json.loads(res.content)
    except Exception:
        return [{"error": "Failed to parse quiz JSON", "raw": res.content}]


# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="AI Study Buddy", version="3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Schemas
# ---------------------------
class RegisterIn(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=6, max_length=128)


class LoginIn(BaseModel):
    username: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class QueryIn(BaseModel):
    question: str = Field(min_length=2)


class QuizQueryIn(BaseModel):
    question: str = Field(min_length=2)
    num_questions: int = Field(default=10, ge=1, le=50, description="Number of quiz questions to generate (1-50)")


# ---------------------------
# Endpoints
# ---------------------------
@app.post("/auth/register", response_model=TokenOut)
def register(payload: RegisterIn, db: Session = Depends(get_db)):
    user = create_user(db, payload.username, payload.password)
    token = create_token(user.username)
    return {"access_token": token, "token_type": "bearer"}


@app.post("/auth/login", response_model=TokenOut)
def login(payload: LoginIn, db: Session = Depends(get_db)):
    user = authenticate_user(db, payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Generate token
    token = create_token(user.username)

    # Auto reset vectorstore for this user
    try:
        vs_path = os.path.join(PERSIST_DIR, f"study_buddy_{user.username}")
        if os.path.exists(vs_path):
            shutil.rmtree(vs_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto reset failed: {str(e)}")

    return {"access_token": token, "token_type": "bearer"}


@app.post("/uploadPdf")
async def upload_pdf(file: UploadFile, user_id: str = Depends(get_current_user)):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        add_pdf_to_db(file_bytes, file.filename, user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index PDF: {str(e)}")
    return {"status": "PDF uploaded and indexed", "user": user_id}


@app.post("/askText")
async def ask_text(query: QueryIn, user_id: str = Depends(get_current_user)):
    try:
        return answer_question(query.question, user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_quiz")
async def generate_quiz(query: QuizQueryIn, user_id: str = Depends(get_current_user)):
    try:
        qa = get_user_qa_chain(user_id)
        result = qa.invoke(query.question)
        context = " ".join([doc.page_content for doc in result["source_documents"]])
        quiz = generate_quiz_from_context(context, query.num_questions)
        return {"quiz": quiz, "sources": [doc.metadata for doc in result["source_documents"]]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))