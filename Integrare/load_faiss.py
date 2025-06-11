import os
import glob
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Cale relativă față de acest fișier
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_FOLDER = os.path.join(BASE_DIR, "output")

URL_MAPPINGS = {
    "admitere": "www.inginerie.ulbsibiu.ro/admitere",
    "inscriere": "www.inginerie.ulbsibiu.ro/admitere",
    "orar": "www.inginerie.ulbsibiu.ro/orar",
    "cursuri": "www.inginerie.ulbsibiu.ro/cursuri",
    "examene": "www.inginerie.ulbsibiu.ro/examene",
    "note": "www.inginerie.ulbsibiu.ro/note",
    "secretariat": "www.inginerie.ulbsibiu.ro/contact",
    "bursă": "www.inginerie.ulbsibiu.ro/burse",
    "cazare": "www.inginerie.ulbsibiu.ro/cazare",
    "cariera": "www.inginerie.ulbsibiu.ro/cariere",
    "practica": "www.inginerie.ulbsibiu.ro/practica",
    "cercetare": "www.inginerie.ulbsibiu.ro/cercetare",
    "evenimente": "www.inginerie.ulbsibiu.ro/evenimente",
    "biblioteca": "www.inginerie.ulbsibiu.ro/biblioteca",
    "erasmus": "www.inginerie.ulbsibiu.ro/erasmus"
}


def load_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())


def load_txt(path):
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()


def load_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".txt":
        return load_txt(path)
    else:
        raise ValueError(f"Unsupported file: {path}")


def load_documents(folder_path):
    pdfs = glob.glob(os.path.join(folder_path, "*.pdf"))
    txts = glob.glob(os.path.join(folder_path, "*.txt"))
    all_files = pdfs + txts
    docs = []

    for path in all_files:
        try:
            text = load_file(path)
            if text.strip():
                filename = os.path.basename(path)
                docs.append(f"[Source: {filename}]\n{text}")
        except Exception as e:
            print(f"⚠ Eroare la fișierul {path}: {e}")
    print(f"[INFO] S-au încărcat {len(docs)} documente din folderul '{folder_path}'.")
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents(docs)


def get_vector_store():
    docs = load_documents(DOCUMENTS_FOLDER)
    if not docs:
        return None
    chunks = chunk_documents(docs)
    texts = [doc.page_content for doc in chunks]
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(texts, embedder)


def find_relevant_urls(question, context=""):
    combined = f"{question} {context}".lower()
    return [url for keyword, url in URL_MAPPINGS.items() if keyword in combined]

if __name__ == '__main__':
    load_documents(DOCUMENTS_FOLDER)
