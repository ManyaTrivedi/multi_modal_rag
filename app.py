import os
import warnings
warnings.filterwarnings("ignore")

try:
    import pymupdf as fitz
except ImportError:
    import fitz
import pdfplumber

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_core.documents import Document
import requests
import base64
import json

app = FastAPI(title="SpectraModal AI - Multi-Modal RAG")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount uploads directory to serve extracted charts, figures, and PDFs directly
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Initialize Local Embeddings & LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_llm = Ollama(model="llama3")

# In-memory document session state
current_vectorstore = None
current_doc_info = {
    "filename": None,
    "total_pages": 0,
    "figures": [],
    "tables_count": 0,
    "chunks_indexed": 0
}

def encode_image(image_path: str) -> str:
    """Helper function to convert image to base64 for LLaVA."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def describe_image_with_llava(image_path: str) -> str:
    """Analyze chart or figure using local LLaVA vision model via Ollama."""
    try:
        base64_image = encode_image(image_path)
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llava",
            "prompt": (
                "You are an expert technical and financial chart analyst. "
                "Analyze this image/figure/chart in detail. Extract all data points, numerical metrics, trends, "
                "legends, axis labels, titles, and text visible so it can be indexed for precise document QA."
            ),
            "images": [base64_image],
            "stream": False
        }
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "No visual analysis returned.")
        else:
            return f"Vision model response code: {response.status_code}"
    except Exception as e:
        return f"Local vision analysis unavailable: {str(e)}"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

@app.get("/doc-info")
async def get_doc_info():
    """Returns metadata and extracted figures for the currently indexed document."""
    global current_doc_info, current_vectorstore
    
    # If no active document in memory, check if uploads directory contains previous files
    if not current_doc_info.get("filename"):
        pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
        if pdf_files:
            pdf_name = pdf_files[0]
            img_files = sorted([
                f for f in os.listdir(UPLOAD_DIR)
                if f.startswith("page_") and f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
            ])
            cached_figs = []
            for img in img_files:
                try:
                    page_num = int(img.split("_")[1])
                except Exception:
                    page_num = 1
                cached_figs.append({
                    "page": page_num,
                    "image_url": f"/uploads/{img}",
                    "filename": img,
                    "description": f"Extracted figure from Page {page_num}"
                })
            return {
                "indexed": current_vectorstore is not None,
                "filename": pdf_name,
                "total_pages": None,
                "figures": cached_figs,
                "tables_count": None,
                "chunks_indexed": None
            }

    return {
        "indexed": current_vectorstore is not None,
        **current_doc_info
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global current_vectorstore, current_doc_info
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    print(f"[*] Processing Multi-Modal PDF: {file.filename}")
    
    documents = []
    total_tables = 0
    total_pages = 0
    extracted_figures = []
    
    # 1. Parse text & tables using pdfplumber
    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            
            # Extract tables
            tables = page.extract_tables()
            table_text = ""
            has_table = False
            if tables:
                has_table = True
                total_tables += len(tables)
                for t_idx, table in enumerate(tables):
                    table_text += f"\n[Table {t_idx+1} on Page {i+1}]\n"
                    for row in table:
                        clean_row = [str(cell).strip() if cell else "" for cell in row]
                        table_text += " | ".join(clean_row) + "\n"
            
            content = f"Page {i+1} Text:\n{text}\n\nTables:\n{table_text}"
            documents.append(Document(
                page_content=content,
                metadata={
                    "source": file.filename,
                    "page": i + 1,
                    "type": "table" if has_table else "text"
                }
            ))

    # 2. Extract Images/Figures using PyMuPDF and describe them with LLaVA
    doc = fitz.open(file_path)
    for i, page in enumerate(doc):
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Skip tiny images / icons (smaller than 60x60 or < 2KB)
            if len(image_bytes) < 2048:
                continue
                
            image_basename = f"page_{i+1}_img_{img_index}.{image_ext}"
            image_filename = os.path.join(UPLOAD_DIR, image_basename)
            with open(image_filename, "wb") as img_file:
                img_file.write(image_bytes)
                
            print(f"[*] Found figure on page {i+1}. Sending to local LLaVA vision model...")
            image_description = describe_image_with_llava(image_filename)
            
            rel_url = f"/uploads/{image_basename}"
            extracted_figures.append({
                "page": i + 1,
                "image_url": rel_url,
                "filename": image_basename,
                "description": image_description
            })
            
            figure_content = f"[Figure/Chart Analysis on Page {i+1}]:\n{image_description}"
            documents.append(Document(
                page_content=figure_content,
                metadata={
                    "source": file.filename,
                    "page": i + 1,
                    "type": "figure",
                    "image_url": rel_url,
                    "description": image_description[:160]
                }
            ))

    # 3. Chunk documents into dense vectors
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    # 4. Store in Chroma Vector DB
    current_vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    current_doc_info = {
        "filename": file.filename,
        "total_pages": total_pages,
        "figures": extracted_figures,
        "tables_count": total_tables,
        "chunks_indexed": len(splits)
    }
    
    return {
        "status": "success",
        "filename": file.filename,
        "chunks_indexed": len(splits),
        "total_pages": total_pages,
        "figures": extracted_figures,
        "tables_count": total_tables
    }

@app.post("/chat")
async def chat_endpoint(request: Request):
    global current_vectorstore
    
    if current_vectorstore is None:
        return {
            "answer": "⚠️ **No document has been indexed yet.** Please upload a multi-modal PDF in the sidebar to start asking questions!",
            "sources": []
        }

    data = await request.json()
    query = data.get("question")
    
    if not query:
        return JSONResponse(status_code=400, content={"error": "No question provided"})

    # Retrieve relevant text, tables, and figure descriptions
    retriever = current_vectorstore.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke(query)
    
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Build structured sources with image URLs and previews
    seen_keys = set()
    sources = []
    for doc in retrieved_docs:
        source_name = doc.metadata.get("source", "Document")
        page = doc.metadata.get("page", 1)
        doc_type = doc.metadata.get("type", "text")
        image_url = doc.metadata.get("image_url", None)
        
        key = f"{source_name}_{page}_{doc_type}_{image_url}"
        if key not in seen_keys:
            seen_keys.add(key)
            snippet = doc.page_content.strip()
            # Truncate clean snippet
            snippet_preview = snippet[:220].replace("\n", " ") + ("..." if len(snippet) > 220 else "")
            sources.append({
                "source": source_name,
                "page": page,
                "type": doc_type,
                "image_url": image_url,
                "preview": snippet_preview
            })

    prompt = f"""You are SpectraModal AI, an elite multi-modal document intelligence system.
Answer the user's question with exceptional accuracy, depth, and clarity using ONLY the context below.
The context contains extracted page text, data tables, and vision AI analysis of figures/charts.

Formatting Rules:
1. Use rich Markdown (bolding, clear section headings, structured bullet points).
2. If data is comparative, organize it in a clean Markdown table.
3. Explicitly cite specific figures and pages when discussing charts (e.g. "[Figure on Page X]").
4. Explicitly cite tables and pages when referencing tabular data.
5. If the context does not contain sufficient details to answer fully, state what is known and specify what is missing.

Context:
{context}

Question: {query}
Answer:"""

    try:
        answer = text_llm.invoke(prompt)
    except Exception as e:
        answer = f"Error generating response from local LLM: {str(e)}"

    return {
        "answer": answer,
        "sources": sources
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8002, reload=True)