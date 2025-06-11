import os
import glob
import re
from flask import Flask, request, render_template_string
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = Flask(__name__)

# Configuration - Set your folder path here
DOCUMENTS_FOLDER = "D:\chatbot\output"  # Change this to your folder path

# Global variables to store the processed data
vector_store = None
processed_files = []

# URL mapping for different topics/keywords
URL_MAPPINGS = {
    # Admissions related
    "admitere": "www.inginerie.ulbsibiu.ro/admitere",
    "inscriere": "www.inginerie.ulbsibiu.ro/admitere",
    "facultate": "www.inginerie.ulbsibiu.ro/admitere",
    "aplicare": "www.inginerie.ulbsibiu.ro/admitere",

    # Academic information
    "orar": "www.inginerie.ulbsibiu.ro/orar",
    "programa": "www.inginerie.ulbsibiu.ro/programe",
    "cursuri": "www.inginerie.ulbsibiu.ro/cursuri",
    "discipline": "www.inginerie.ulbsibiu.ro/cursuri",
    "materii": "www.inginerie.ulbsibiu.ro/cursuri",

    # Exams and grades
    "examene": "www.inginerie.ulbsibiu.ro/examene",
    "note": "www.inginerie.ulbsibiu.ro/note",
    "restante": "www.inginerie.ulbsibiu.ro/examene",
    "sesiune": "www.inginerie.ulbsibiu.ro/examene",

    # Administrative
    "secretariat": "www.inginerie.ulbsibiu.ro/contact",
    "contact": "www.inginerie.ulbsibiu.ro/contact",
    "telefon": "www.inginerie.ulbsibiu.ro/contact",
    "email": "www.inginerie.ulbsibiu.ro/contact",

    # Student services
    "bursă": "www.inginerie.ulbsibiu.ro/burse",
    "burse": "www.inginerie.ulbsibiu.ro/burse",
    "cazare": "www.inginerie.ulbsibiu.ro/cazare",
    "camin": "www.inginerie.ulbsibiu.ro/cazare",

    # Career and opportunities
    "cariera": "www.inginerie.ulbsibiu.ro/cariere",
    "job": "www.inginerie.ulbsibiu.ro/cariere",
    "locuri de munca": "www.inginerie.ulbsibiu.ro/cariere",
    "practica": "www.inginerie.ulbsibiu.ro/practica",
    "stagiu": "www.inginerie.ulbsibiu.ro/practica",

    # Research and projects
    "cercetare": "www.inginerie.ulbsibiu.ro/cercetare",
    "proiecte": "www.inginerie.ulbsibiu.ro/proiecte",
    "laborator": "www.inginerie.ulbsibiu.ro/laboratoare",

    # Events and news
    "evenimente": "www.inginerie.ulbsibiu.ro/evenimente",
    "noutati": "www.inginerie.ulbsibiu.ro/noutati",
    "stiri": "www.inginerie.ulbsibiu.ro/noutati",

    # Library and resources
    "biblioteca": "www.inginerie.ulbsibiu.ro/biblioteca",
    "carti": "www.inginerie.ulbsibiu.ro/biblioteca",
    "resurse": "www.inginerie.ulbsibiu.ro/resurse",

    # Add more mappings as needed
    "erasmus": "www.inginerie.ulbsibiu.ro/erasmus",
    "mobilitate": "www.inginerie.ulbsibiu.ro/erasmus",
    "international": "www.inginerie.ulbsibiu.ro/international"
}

# Load TinyLlama model once
print("Loading model...")

base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "D:/chatbot/adapter"

print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id).to("cuda")

print("Loading PEFT adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)

# Combinează adapterul cu modelul de bază într-un model final
print("Merging adapter with base model...")
model = model.merge_and_unload().to("cuda")

# Acum poți folosi pipeline-ul
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, max_new_tokens=256)
print("Model with adapter merged and ready.")

print("Model loaded on GPU.")


# Load and extract text from PDF
def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text


# Load and extract text from TXT file
def load_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()


# General file loader that handles both PDF and TXT
def load_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        return load_pdf(file_path)
    elif file_extension == '.txt':
        return load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


# Load all documents from the folder
def load_documents_from_folder(folder_path):
    """Load all PDF and TXT files from the specified folder"""
    global processed_files

    if not os.path.exists(folder_path):
        print(f"Warning: Folder '{folder_path}' does not exist!")
        return []

    # Get all PDF and TXT files
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    all_files = pdf_files + txt_files

    if not all_files:
        print(f"No PDF or TXT files found in '{folder_path}'")
        return []

    documents = []
    processed_files = []

    print(f"Found {len(all_files)} files to process...")

    for file_path in all_files:
        try:
            filename = os.path.basename(file_path)
            print(f"Processing: {filename}")

            text = load_file(file_path)

            if text.strip():
                # Add filename info to the text for better context
                text_with_source = f"[Source: {filename}]\n{text}"
                documents.append(text_with_source)
                processed_files.append(filename)
                print(f"✓ Successfully processed: {filename}")
            else:
                print(f"⚠ Skipped empty file: {filename}")

        except Exception as e:
            print(f"✗ Error processing {filename}: {str(e)}")

    print(f"Successfully processed {len(documents)} files")
    return documents


# Split text into chunks
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents([text])


# Build FAISS vector store from all documents
def create_vector_store_from_folder():
    """Create vector store from all documents in the folder"""
    global vector_store

    # Load all documents
    documents = load_documents_from_folder(DOCUMENTS_FOLDER)

    if not documents:
        print("No documents to process!")
        return False

    # Combine all documents
    all_text = "\n\n".join(documents)

    # Create chunks
    chunks = chunk_text(all_text)

    # Create vector store
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    texts = [doc.page_content for doc in chunks]
    vector_store = FAISS.from_texts(texts, embedder)

    print("Vector store created successfully!")
    return True


# Find relevant URLs based on question content
def find_relevant_urls(question, context=""):
    """Find URLs that match keywords in the question or context"""
    relevant_urls = []
    question_lower = question.lower()
    context_lower = context.lower()
    combined_text = f"{question_lower} {context_lower}"

    # Check for keyword matches
    for keyword, url in URL_MAPPINGS.items():
        if keyword in combined_text:
            if url not in relevant_urls:
                relevant_urls.append(url)

    return relevant_urls


# Generate answer using TinyLLaMA with URLs
def generate_answer_with_urls(context, question):
    prompt = f"""<|user|>\nContext:\n{context}\n\nÎntrebare: {question}\nTe rog să răspunzi în limba română.\n<|assistant|>"""
    outputs = pipe(prompt)
    answer = outputs[0]['generated_text'].replace(prompt, "").strip()

    # Find relevant URLs
    relevant_urls = find_relevant_urls(question, context)

    # Add URLs to the answer if found
    if relevant_urls:
        url_text = "\n\nMai multe informații:"
        for i, url in enumerate(relevant_urls[:3], 1):  # Limit to 3 URLs
            url_text += f"\n• {url}"
        answer += url_text

    return answer


HTML_TEMPLATE = """
<!doctype html>
<html lang="ro">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Asistentul tău cu documente</title>
  <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --main-blue: #0057b8;
      --light-blue: #e6f0ff;
      --gray-bg: #f9f9f9;
      --gray-border: #d0d0d0;
      --text-color: #222;
      --link-color: #0066cc;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      padding: 0;
      font-family: 'Roboto Mono', monospace;
      background: var(--gray-bg);
      color: var(--text-color);
      line-height: 1.6;
    }

    .container {
      max-width: 680px;
      margin: 40px auto;
      background: #fff;
      padding: 30px;
      border-radius: 8px;
      border: 1px solid var(--gray-border);
    }

    h1 {
      text-align: center;
      color: var(--main-blue);
      font-size: 24px;
      margin-bottom: 24px;
    }

    input[type="text"] {
      width: 100%;
      padding: 14px;
      margin: 20px 0;
      border: 1px solid var(--gray-border);
      border-radius: 6px;
      font-size: 16px;
      background: #fefefe;
    }

    input[type="submit"] {
      width: 100%;
      background-color: var(--main-blue);
      color: white;
      padding: 14px;
      border: none;
      border-radius: 6px;
      font-size: 16px;
      font-weight: bold;
      transition: background-color 0.3s ease;
      cursor: pointer;
    }

    input[type="submit"]:hover {
      background-color: #004494;
    }

    .answer-box, .error-box {
      padding: 20px;
      margin-top: 30px;
      border-left: 4px solid;
      border-radius: 6px;
      background: var(--light-blue);
    }

    .answer-box {
      border-color: var(--main-blue);
    }

    .error-box {
      background: #ffe6e6;
      border-color: #cc0000;
      color: #800000;
    }

    h2 {
      margin-top: 0;
      font-size: 18px;
    }

    .answer-text {
      white-space: pre-line;
      margin-bottom: 15px;
    }

    .url-links {
      background: #f0f8ff;
      padding: 15px;
      border-radius: 4px;
      border: 1px solid #d0e7ff;
      margin-top: 15px;
    }

    .url-links h3 {
      margin: 0 0 10px 0;
      font-size: 14px;
      color: var(--main-blue);
      font-weight: 600;
    }

    .url-links a {
      color: var(--link-color);
      text-decoration: none;
      display: block;
      margin: 8px 0;
      padding: 5px 0;
      border-bottom: 1px dotted #ccc;
    }

    .url-links a:hover {
      color: #004488;
      text-decoration: underline;
    }

    .url-links a:last-child {
      border-bottom: none;
    }

    .footer {
      text-align: center;
      margin-top: 40px;
      font-size: 12px;
      color: #999;
    }

    @media (max-width: 768px) {
      .container {
        margin: 20px;
        padding: 20px;
      }

      h1 {
        font-size: 20px;
      }

      input[type="text"], input[type="submit"] {
        font-size: 15px;
        padding: 12px;
      }

      h2 {
        font-size: 16px;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Asistentul tău cu documente 📄</h1>

    {% if vector_store_ready %}
      <form method="post">
        <input type="text" name="question" placeholder="Pune o întrebare..." required>
        <input type="submit" value="Răspunde">
      </form>
    {% else %}
      <div class="error-box">
        <strong>⚠ Documentele nu sunt încărcate.</strong>
        <p>Verifică dosarul documentelor și reîncarcă aplicația.</p>
      </div>
    {% endif %}

    {% if answer %}
      <div class="answer-box">
        <h2>💬 Răspuns:</h2>
        <div class="answer-text">{{ answer }}</div>
      </div>
    {% endif %}

    {% if error %}
      <div class="error-box">
        <h2>❌ Eroare:</h2>
        <p>{{ error }}</p>
      </div>
    {% endif %}

    <div class="footer">
      <p>Disclaimer - raspunusurile sunt generate automat fara garantie de corectitudine</p>
      <p>Pentru mai multe detalii acessati site-ul faculatii la www.inginerie.ulbsibiu.ro</p>
    </div>
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    global vector_store, processed_files

    answer = None
    error = None

    # Check if refresh is requested
    if request.args.get('refresh') == '1':
        print("Refreshing documents...")
        create_vector_store_from_folder()

    # Initialize vector store if not already done
    if vector_store is None:
        print("Initializing vector store...")
        create_vector_store_from_folder()

    if request.method == "POST":
        question = request.form.get("question")

        if not question:
            error = "Te rog să introduci o întrebare."
        elif vector_store is None:
            error = "Nu există documente încărcate. Te rog să verifici dosarul documentelor."
        else:
            try:
                # Search for relevant documents
                relevant_docs = vector_store.similarity_search(question, k=3)
                context = "\n".join([doc.page_content for doc in relevant_docs])

                if not context.strip():
                    error = "Nu s-au găsit informații relevante în documentele disponibile."
                else:
                    # Generate answer with URLs
                    answer = generate_answer_with_urls(context, question)

            except Exception as e:
                error = f"A apărut o eroare la procesarea întrebării: {str(e)}"

    return render_template_string(HTML_TEMPLATE,
                                  answer=answer,
                                  error=error,
                                  folder_path=DOCUMENTS_FOLDER,
                                  file_count=len(processed_files),
                                  processed_files=processed_files,
                                  vector_store_ready=vector_store is not None)


if __name__ == "__main__":
    print(f"Document folder set to: {DOCUMENTS_FOLDER}")
    print("Starting Flask application...")
    app.run(debug=True)