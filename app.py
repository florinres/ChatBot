import os
from flask import Flask, request, render_template_string
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

app = Flask(__name__)

# Load TinyLlama model once
print("Loading model...")
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
print("Model loaded.")

# Load and extract text from PDF
def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

# Split text into chunks
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents([text])

# Build FAISS vector store
def create_vector_store(chunks):
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    texts = [doc.page_content for doc in chunks]
    return FAISS.from_texts(texts, embedder)

# Generate answer using TinyLLaMA
def generate_answer(context, question):
    prompt = f"""<|user|>\nContext:\n{context}\n\nQuestion: {question}\n<|assistant|>"""
    outputs = pipe(prompt)
    answer = outputs[0]['generated_text'].replace(prompt, "").strip()
    return answer

# HTML Template for upload + question form + result display
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>PDF Q&A with TinyLLaMA</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #f4f6f9;
      color: #333;
      padding: 40px;
    }
    .container {
      max-width: 700px;
      margin: auto;
      background: white;
      padding: 30px 40px;
      border-radius: 12px;
      box-shadow: 0 0 20px rgba(0,0,0,0.05);
    }
    h1 {
      text-align: center;
      color: #007BFF;
      margin-bottom: 30px;
    }
    input[type="file"],
    input[type="text"] {
      width: 100%;
      padding: 12px;
      margin: 12px 0;
      border: 1px solid #ccc;
      border-radius: 6px;
      box-sizing: border-box;
    }
    input[type="submit"] {
      width: 100%;
      background-color: #007BFF;
      color: white;
      padding: 14px;
      margin-top: 16px;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      font-size: 16px;
    }
    input[type="submit"]:hover {
      background-color: #0056b3;
    }
    .answer-box {
      background: #e9f5ff;
      padding: 20px;
      margin-top: 30px;
      border-left: 4px solid #007BFF;
      border-radius: 6px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>PDF Q&A with TinyLLaMA</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="pdf_file" required>
      <input type="text" name="question" placeholder="Enter your question here..." required>
      <input type="submit" value="Get Answer">
    </form>

    {% if answer %}
      <div class="answer-box">
        <h2>💡 Answer:</h2>
        <p>{{ answer }}</p>
      </div>
    {% endif %}
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    if request.method == "POST":
        pdf_file = request.files.get("pdf_file")
        question = request.form.get("question")

        if not pdf_file or not question:
            answer = "Please upload a PDF and enter a question."
        else:
            # Save PDF temporarily
            pdf_path = f"./temp_{pdf_file.filename}"
            pdf_file.save(pdf_path)

            # Process PDF
            text = load_pdf(pdf_path)
            chunks = chunk_text(text)
            vector_store = create_vector_store(chunks)
            relevant_docs = vector_store.similarity_search(question, k=3)
            context = "\n".join([doc.page_content for doc in relevant_docs])

            # Generate answer
            answer = generate_answer(context, question)

            # Cleanup temp file
            os.remove(pdf_path)

    return render_template_string(HTML_TEMPLATE, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
