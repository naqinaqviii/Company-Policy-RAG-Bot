# app.py
# Company Policy RAG + Streamlit Dashboard (free, from-scratch, full code)

import os
import io
import sqlite3
from datetime import datetime
from typing import List, Tuple

import numpy as np
import streamlit as st
from passlib.hash import bcrypt

# PDF/text handling
import PyPDF2
try:
    import fitz  # PyMuPDF (optional, for better previews)
    FITZ_AVAILABLE = True
except Exception:
    FITZ_AVAILABLE = False

# Embeddings + generator
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ------------------ Configuration ------------------
DATA_DIR = "data"
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
DB_PATH = os.path.join(DATA_DIR, "company.db")

EMBED_MODEL = "all-MiniLM-L6-v2"        # small, fast, free
GEN_MODEL = "google/flan-t5-small"      # free text2text model
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# Out-of-scope (OOS) threshold — if top cosine similarity < this, we say "not found"
OOS_THRESHOLD = 0.35

LOGO_PATH = "infotech.png"  # place this file next to app.py

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ------------------ DB Helpers ------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()

    # Users (employees)
    c.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        department TEXT,
        is_admin INTEGER DEFAULT 0
    );
    """)

    # Query history
    c.execute("""
    CREATE TABLE IF NOT EXISTS queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id INTEGER,
        question TEXT,
        answer TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (employee_id) REFERENCES employees (id)
    );
    """)

    # Files
    c.execute("""
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        stored_path TEXT NOT NULL,
        size_bytes INTEGER,
        pages INTEGER,
        filetype TEXT,
        uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Chunks with embeddings
    c.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL,
        content TEXT NOT NULL,
        embedding BLOB NOT NULL,
        source TEXT,
        FOREIGN KEY (file_id) REFERENCES files (id) ON DELETE CASCADE
    );
    """)

    conn.commit()
    conn.close()


# ------------------ Auth Helpers ------------------
def hash_password(password: str) -> str:
    return bcrypt.hash(password)

def check_password(password: str, hashed: str) -> bool:
    return bcrypt.verify(password, hashed)


def create_user(name, email, password, department, is_admin=0):
    conn = get_conn()
    c = conn.cursor()
    pw_hash = hash_password(password)
    try:
        c.execute(
            "INSERT INTO employees (name, email, password_hash, department, is_admin) VALUES (?, ?, ?, ?, ?)",
            (name, email, pw_hash, department, is_admin)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user_by_email(email):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT id, name, email, password_hash, department, is_admin FROM employees WHERE email = ?",
        (email,)
    )
    row = c.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "name": row[1], "email": row[2],
                "password_hash": row[3], "department": row[4], "is_admin": row[5]}
    return None


# ------------------ Document Store ------------------
def _normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / (n + 1e-12)

class DocumentStore:
    def __init__(self, embed_model: str = EMBED_MODEL):
        self.embedder = SentenceTransformer(embed_model)

    # --- File handling ---
    def add_file(self, file_bytes: bytes, filename: str, filetype: str) -> Tuple[int, int]:
        """Save uploaded file to disk, insert into files table, chunk + embed into chunks."""
        # store file
        stored_path = os.path.join(UPLOAD_DIR, f"{int(datetime.utcnow().timestamp())}_{filename}")
        with open(stored_path, "wb") as f:
            f.write(file_bytes)

        # meta
        size_bytes = len(file_bytes)
        pages = None
        if filetype == "pdf":
            pages = self._count_pdf_pages(file_bytes)

        conn = get_conn()
        c = conn.cursor()
        c.execute("""
            INSERT INTO files (filename, stored_path, size_bytes, pages, filetype)
            VALUES (?, ?, ?, ?, ?)
        """, (filename, stored_path, size_bytes, pages, filetype))
        file_id = c.lastrowid
        conn.commit()

        # extract text and chunk
        text = ""
        if filetype == "pdf":
            text = self._extract_text_pdf(file_bytes)
        else:
            try:
                text = file_bytes.decode("utf-8", errors="ignore")
            except Exception:
                text = ""

        chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        count_chunks = 0
        if chunks:
            embeddings = self.embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
            embeddings = np.array([_normalize(e).astype(np.float32) for e in embeddings])
            for content, emb in zip(chunks, embeddings):
                c.execute("""
                    INSERT INTO chunks (file_id, content, embedding, source)
                    VALUES (?, ?, ?, ?)
                """, (file_id, content, emb.tobytes(), filename))
                count_chunks += 1
            conn.commit()

        conn.close()
        return file_id, count_chunks

    def list_files(self):
        conn = get_conn()
        c = conn.cursor()
        c.execute("""
            SELECT f.id, f.filename, f.stored_path, f.size_bytes, f.pages, f.filetype, f.uploaded_at,
                   COUNT(ch.id) as chunk_count
            FROM files f
            LEFT JOIN chunks ch ON ch.file_id = f.id
            GROUP BY f.id
            ORDER BY f.uploaded_at DESC
        """)
        rows = c.fetchall()
        conn.close()
        files = []
        for r in rows:
            files.append({
                "id": r[0], "filename": r[1], "stored_path": r[2], "size": r[3],
                "pages": r[4], "filetype": r[5], "uploaded_at": r[6], "chunks": r[7]
            })
        return files

    def delete_file(self, file_id: int):
        conn = get_conn()
        c = conn.cursor()
        c.execute("SELECT stored_path FROM files WHERE id = ?", (file_id,))
        row = c.fetchone()
        path = row[0] if row else None

        # delete rows (thanks to ON DELETE CASCADE on chunks)
        c.execute("DELETE FROM files WHERE id = ?", (file_id,))
        conn.commit()
        conn.close()

        # remove from disk
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

    def clear_index(self):
        conn = get_conn()
        c = conn.cursor()
        c.execute("DELETE FROM chunks;")
        c.execute("DELETE FROM files;")
        conn.commit()
        conn.close()

    def preview_file_text(self, file_id: int, max_chars: int = 1500) -> str:
        """Concatenate first few chunks for preview."""
        conn = get_conn()
        c = conn.cursor()
        c.execute("SELECT content FROM chunks WHERE file_id = ? ORDER BY id ASC LIMIT 20;", (file_id,))
        parts = [row[0] for row in c.fetchall()]
        conn.close()
        preview = " ".join(parts)
        return preview[:max_chars] + ("..." if len(preview) > max_chars else "")

    # --- Retrieval ---
    def num_chunks(self) -> int:
        conn = get_conn()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM chunks;")
        n = c.fetchone()[0]
        conn.close()
        return n

    def retrieve(self, query: str, top_k: int = 3):
        """Return list of dicts: {content, source, file_id, score} where score is cosine similarity."""
        if self.num_chunks() == 0:
            return []

        q_emb = self.embedder.encode([query], convert_to_numpy=True)[0]
        q_emb = _normalize(q_emb.astype(np.float32))

        conn = get_conn()
        c = conn.cursor()
        c.execute("SELECT id, file_id, content, embedding, source FROM chunks;")
        rows = c.fetchall()
        conn.close()

        contents = []
        embs = []
        metas = []
        for row in rows:
            cid, file_id, content, emb_blob, source = row
            emb_vec = np.frombuffer(emb_blob, dtype=np.float32)
            contents.append(content)
            embs.append(emb_vec)
            metas.append((cid, file_id, source))

        embs = np.vstack(embs)  # (N, d)
        # cosine similarity = dot since all are normalized
        sims = embs @ q_emb
        idxs = np.argsort(-sims)[:min(top_k, sims.shape[0])]

        results = []
        for i in idxs:
            cid, file_id, source = metas[i]
            results.append({
                "content": contents[i],
                "source": source,
                "file_id": file_id,
                "score": float(sims[i])
            })
        return results

    # --- Internal: PDF helpers ---
    def _count_pdf_pages(self, file_bytes: bytes) -> int:
        try:
            if FITZ_AVAILABLE:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                pages = doc.page_count
                doc.close()
                return pages
            else:
                reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                return len(reader.pages)
        except Exception:
            return None

    def _extract_text_pdf(self, file_bytes: bytes) -> str:
        try:
            if FITZ_AVAILABLE:
                text_parts = []
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                for page in doc:
                    text_parts.append(page.get_text())
                doc.close()
                return "\n".join(text_parts)
            else:
                reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                full = []
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        full.append(t)
                return "\n".join(full)
        except Exception:
            return ""


# ------------------ Embedding / Generation ------------------
@st.cache_resource
def load_docstore():
    return DocumentStore(EMBED_MODEL)

@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model=GEN_MODEL)


# ------------------ Utilities ------------------
def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    t = " ".join(text.split())
    if not t:
        return []
    out = []
    start = 0
    while start < len(t):
        end = min(len(t), start + chunk_size)
        out.append(t[start:end])
        start = max(end - overlap, end)  # avoid infinite loop
    return out

def save_query(employee_id: int, question: str, answer: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT INTO queries (employee_id, question, answer, timestamp) VALUES (?, ?, ?, ?)",
              (employee_id, question, answer, datetime.utcnow()))
    conn.commit()
    conn.close()


# ------------------ RAG Answer ------------------
docstore = load_docstore()
generator = load_generator()

def rag_answer(question: str, top_k: int = 3) -> Tuple[str, List[dict], float]:
    """Returns (answer_text, contexts, top_score). Applies OOS_THRESHOLD."""
    contexts = docstore.retrieve(question, top_k=top_k)
    top_score = contexts[0]["score"] if contexts else 0.0

    # Out-of-scope guard
    if not contexts or top_score < OOS_THRESHOLD:
        msg = ("Sorry — I couldn't find any relevant policy information for your question in the uploaded documents. "
               "Please rephrase or upload the relevant policy.")
        return msg, contexts, top_score

    context_text = "\n\n".join([c["content"] for c in contexts])
    prompt = (
        "Use ONLY the following company policy excerpts to answer the question. "
        "If the answer is not present, say you don't know.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\nAnswer:"
    )
    out = generator(prompt, max_length=256, do_sample=False)
    answer = out[0]["generated_text"]
    return answer, contexts, top_score


# ------------------ Theming & Header ------------------
st.set_page_config(page_title="Company Policy RAG Assistant", layout="wide")

# Green + Blue theme + thicker headings + nicer buttons
st.markdown("""
<style>
/* Headings thicker, blue */
h1, h2, h3 {
    font-weight: 800 !important;
    letter-spacing: .3px;
    color: #1E88E5 !important; /* blue */
}

/* File uploader card */
[data-testid="stFileUploader"] {
    background-color: #E8F5E9;      /* very light green */
    border: 2px dashed #1E88E5;     /* blue border */
    padding: 12px;
    border-radius: 12px;
}

/* Buttons (green), hover (blue) */
button[kind="secondary"], .stButton button, .stDownloadButton button {
    background-color: #43A047 !important; /* green */
    color: white !important;
    border-radius: 8px !important;
    border: 0 !important;
}
button[kind="secondary"]:hover, .stButton button:hover, .stDownloadButton button:hover {
    background-color: #1E88E5 !important; /* blue */
    color: #fff !important;
}

/* Info/success cards rounded */
.stAlert {
    border-radius: 10px !important;
}

/* Tables text */
[data-testid="stTable"] td, [data-testid="stTable"] th {
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

def page_header(title: str):
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title(title)
    with col2:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)


# ------------------ App ------------------
import streamlit as st

# Gradient navbar + dark theme styling
st.markdown(
    """
    <style>
    /* Page Background */
    .main {
        background-color: #0F172A;
        padding: 0;
    }

    /* Gradient Navbar */
    .navbar {
        background: linear-gradient(90deg, #0A192F 0%, #1E3A8A 100%);
        padding: 20px 40px;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .navbar h1 {
        color: #E2E8F0;
        font-size: 28px;
        font-weight: 800;
        margin: 0;
    }

    .navbar img {
        height: 40px;
    }

    /* Cards */
    section[data-testid="stSidebar"], .block-container {
        background: #1E293B;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        padding: 20px;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #1E3A8A 0%, #2563EB 100%);
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #2563EB 0%, #3B82F6 100%);
        transform: translateY(-2px);
        box-shadow: 0px 4px 12px rgba(59, 130, 246, 0.5);
    }

    /* Titles */
    h1, h2, h3, label {
        color: #E2E8F0;
    }

    /* Input boxes */
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 1px solid #334155;
        padding: 10px;
        background-color: #0F172A;
        color: #E2E8F0;
    }

    .stTextInput>div>div>input:focus {
        border-color: #3B82F6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Navbar HTML
st.markdown(
    """
    <div class="navbar">
        <h1>🚀 RAG Assistant Dashboard</h1>
        <img src="https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg" alt="Logo">
    </div>
    """,
    unsafe_allow_html=True
)
# Example chat messages
st.markdown(
    """
    <div class="chat-container">
        <div class="user-message">What is the leave policy?</div>
        <div class="bot-message">The company provides 14 annual leaves per year.</div>
        <div class="user-message">Great, how many sick leaves?</div>
        <div class="bot-message">You are allowed 10 sick leaves annually.</div>
    </div>
    """,
    unsafe_allow_html=True
)

def main():
    init_db()

    if "user" not in st.session_state:
        st.session_state["user"] = None

    st.sidebar.title("Navigation")
    if st.session_state["user"]:
        st.sidebar.write(f"Logged in as: **{st.session_state['user']['name']}**")
        if st.sidebar.button("Logout"):
            st.session_state["user"] = None
            st.rerun()
    else:
        st.sidebar.write("Not logged in")

    menu = st.sidebar.radio("Go to", ["Home", "Login / Signup", "Admin (if you are admin)"])

    if menu == "Home":
        page_header("Company Policy — RAG Assistant")
        st.write("A local, free Retrieval-Augmented Generation assistant for company policies. Built from scratch.")
        st.markdown("**Quick steps**: 1) Create an account (first user becomes admin). "
                    "2) Admin uploads policies. 3) Employees ask questions and see answers with sources.")

        if st.session_state["user"]:
            user = st.session_state["user"]
            st.header(f"Welcome, {user['name']}")

            st.subheader("Ask the Policy Bot")
            question = st.text_input("Enter your question about company policies:")
            if st.button("Ask"):
                if docstore.num_chunks() == 0:
                    st.warning("No documents indexed yet. Ask your admin to upload policies.")
                elif question.strip():
                    with st.spinner("Retrieving relevant policies…"):
                        answer, contexts, top_score = rag_answer(question, top_k=3)
                        st.markdown("**Answer:**")
                        st.write(answer)
                        st.caption(f"Top similarity: {top_score:.2f}")
                        if contexts:
                            with st.expander("Show retrieved excerpts (sources)"):
                                for i, c in enumerate(contexts, 1):
                                    st.markdown(f"**{i}. Source:** {c['source']}  |  **score:** {c['score']:.2f}")
                                    st.write(c["content"])
                                    st.markdown("---")
                        save_query(user["id"], question, answer)
                else:
                    st.info("Type a question first.")

            st.subheader("Your query history")
            conn = get_conn()
            c = conn.cursor()
            c.execute("""
                SELECT question, answer, timestamp
                FROM queries
                WHERE employee_id = ?
                ORDER BY timestamp DESC
            """, (user["id"],))
            rows = c.fetchall()
            conn.close()
            if rows:
                for q, a, t in rows:
                    st.write(f"**Q:** {q}")
                    st.write(f"**A:** {a}")
                    st.caption(f"_at {t}_")
                    st.markdown("---")
            else:
                st.write("No queries yet.")
        else:
            st.info("Please login or sign up to access your dashboard.")

    elif menu == "Login / Signup":
        page_header("Login or Create Account")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Login")
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
                if submitted:
                    user = get_user_by_email(email)
                    if user and check_password(password, user["password_hash"]):
                        st.success("Logged in")
                        st.session_state["user"] = {
                            "id": user["id"], "name": user["name"], "email": user["email"], "is_admin": user["is_admin"]
                        }
                        st.rerun()
                    else:
                        st.error("Invalid credentials")

        with col2:
            st.subheader("Sign up")
            with st.form("signup_form"):
                name = st.text_input("Full name")
                email_s = st.text_input("Email (for signup)")
                password_s = st.text_input("Password", type="password")
                dept = st.text_input("Department (optional)")
                submitted2 = st.form_submit_button("Create account")
                if submitted2:
                    conn = get_conn()
                    c = conn.cursor()
                    c.execute("SELECT COUNT(*) FROM employees")
                    count = c.fetchone()[0]
                    conn.close()
                    is_admin = 1 if count == 0 else 0
                    success = create_user(name, email_s, password_s, dept, is_admin)
                    if success:
                        if is_admin:
                            st.success("Admin account created. You are the first user and admin.")
                        else:
                            st.success("Account created. Please login from the left panel.")
                    else:
                        st.error("Failed — email may already exist.")

    elif menu == "Admin (if you are admin)":
        page_header("Admin Panel")
        if not st.session_state["user"] or st.session_state["user"].get("is_admin") != 1:
            st.error("Admin only — please login with an admin account.")
            return

        st.success("Admin access")

        # ---- Upload ----
        st.subheader("Upload policy documents (PDF or TXT)")
        uploaded = st.file_uploader("Upload PDF/TXT", type=["pdf", "txt"], accept_multiple_files=True)
        if uploaded:
            total_chunks = 0
            for file in uploaded:
                filetype = "pdf" if file.name.lower().endswith(".pdf") else "txt"
                file_bytes = file.getvalue()
                file_id, n_chunks = docstore.add_file(file_bytes, file.name, filetype)
                total_chunks += n_chunks
            st.success(f"Uploaded {len(uploaded)} file(s) • Added {total_chunks} chunk(s) • Index updated.")

        st.markdown("---")

        # ---- Files table / view / delete ----
        st.subheader("Your uploaded files")
        files = docstore.list_files()
        if not files:
            st.info("No files uploaded yet.")
        else:
            for f in files:
                with st.expander(f"📄 {f['filename']}  •  {f['filetype'].upper()}  •  chunks: {f['chunks']}  •  uploaded: {f['uploaded_at']}"):
                    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
                    with c1:
                        st.write(f"**File ID:** {f['id']}")
                        st.write(f"**Pages:** {f['pages'] if f['pages'] is not None else '-'}")
                    with c2:
                        kb = (f['size'] or 0) / 1024.0
                        st.write(f"**Size:** {kb:.1f} KB")
                        st.write(f"**Path:** {f['stored_path']}")
                    with c3:
                        if os.path.exists(f["stored_path"]):
                            with open(f["stored_path"], "rb") as fh:
                                st.download_button(
                                    "Download",
                                    data=fh.read(),
                                    file_name=f["filename"],
                                    key=f"download_{f['id']}"   # 🔑 fix here
                                )
                    with c4:
                        if st.button("🗑 Delete this file", key=f"del_{f['id']}"):
                            docstore.delete_file(f["id"])
                            st.warning(f"Deleted {f['filename']} and its chunks.")
                            st.rerun()

                    st.markdown("**Preview:**")
                    preview = docstore.preview_file_text(f["id"], max_chars=1500)
                    st.write(preview if preview else "_(no preview text found)_")

        st.markdown("---")

        # ---- Stats & admin actions ----
        st.subheader("Document store stats")
        st.write(f"**Total chunks stored:** {docstore.num_chunks()}")

        st.subheader("Administrative: create sample policies & reset index")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Create demo policies (sample)"):
                demo_docs = [
                    "Employees are entitled to 14 annual leaves per year.",
                    "Employees can take up to 10 sick leaves per year with a medical certificate.",
                    "Late arrival rules: If you are late more than 3 times in a month, HR will issue a warning.",
                    "Remote work policy: Employees can request remote days with manager approval."
                ]
                for i, d in enumerate(demo_docs, 1):
                    # treat as a 'txt file'
                    filename = f"demo_policy_{i}.txt"
                    docstore.add_file(d.encode("utf-8"), filename, "txt")
                st.success("Demo policies added.")
        with c2:
            if st.button("Reset/clear document store (danger)"):
                docstore.clear_index()
                st.warning("Document store cleared (files + chunks).")

# ---- Run ----
if __name__ == "__main__":
    main()
