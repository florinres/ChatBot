import os
import glob
from datetime import datetime
from flask import Flask, request, render_template_string, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'  # Change this to a random secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'mariadb+mariadbconnector://root:1234@127.0.0.1:3306/chatbot_users'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Configuration - Set your folder path here
DOCUMENTS_FOLDER = "D:\chatbot\output"  # Change this to your folder path

# Global variables to store the processed data
vector_store = None
processed_files = []


# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    chats = db.relationship('Chat', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


# Load TinyLlama model once
print("Loading model...")
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, max_new_tokens=256)
print("Model loaded on GPU.")


# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


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


# Generate answer using TinyLLaMA
def generate_answer(context, question):
    prompt = f"""<|user|>\nContext:\n{context}\n\nÎntrebare: {question}\nTe rog să răspunzi în limba română.\n<|assistant|>"""
    outputs = pipe(prompt)
    answer = outputs[0]['generated_text'].replace(prompt, "").strip()
    return answer


# HTML Templates
LOGIN_TEMPLATE = """
<!doctype html>
<html lang="ro">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Login - Asistentul tău cu documente</title>
  <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --main-blue: #0057b8;
      --light-blue: #e6f0ff;
      --gray-bg: #f9f9f9;
      --gray-border: #d0d0d0;
      --text-color: #222;
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
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
    }

    .container {
      max-width: 400px;
      background: #fff;
      padding: 30px;
      border-radius: 8px;
      border: 1px solid var(--gray-border);
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }

    h1 {
      text-align: center;
      color: var(--main-blue);
      font-size: 24px;
      margin-bottom: 24px;
    }

    .form-group {
      margin-bottom: 20px;
    }

    label {
      display: block;
      margin-bottom: 5px;
      font-weight: 600;
    }

    input[type="text"], input[type="email"], input[type="password"] {
      width: 100%;
      padding: 12px;
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

    .link {
      text-align: center;
      margin-top: 20px;
    }

    .link a {
      color: var(--main-blue);
      text-decoration: none;
    }

    .link a:hover {
      text-decoration: underline;
    }

    .flash-messages {
      margin-bottom: 20px;
    }

    .flash-error {
      background: #ffe6e6;
      color: #800000;
      padding: 10px;
      border-radius: 6px;
      border-left: 4px solid #cc0000;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>{{ title }}</h1>

    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="flash-messages">
          {% for message in messages %}
            <div class="flash-error">{{ message }}</div>
          {% endfor %}
        </div>
      {% endif %}
    {% endwith %}

    <form method="post">
      {% if is_register %}
        <div class="form-group">
          <label for="username">Nume utilizator:</label>
          <input type="text" id="username" name="username" required>
        </div>
        <div class="form-group">
          <label for="email">Email:</label>
          <input type="email" id="email" name="email" required>
        </div>
      {% else %}
        <div class="form-group">
          <label for="username">Nume utilizator sau Email:</label>
          <input type="text" id="username" name="username" required>
        </div>
      {% endif %}

      <div class="form-group">
        <label for="password">Parolă:</label>
        <input type="password" id="password" name="password" required>
      </div>

      <input type="submit" value="{{ submit_text }}">
    </form>

    <div class="link">
      {% if is_register %}
        <a href="{{ url_for('login') }}">Ai deja cont? Loghează-te</a>
      {% else %}
        <a href="{{ url_for('register') }}">Nu ai cont? Înregistrează-te</a>
      {% endif %}
    </div>
  </div>
</body>
</html>
"""

CHAT_TEMPLATE = """
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

    .header {
      background: var(--main-blue);
      color: white;
      padding: 15px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .header h1 {
      margin: 0;
      font-size: 20px;
    }

    .header .user-info {
      display: flex;
      align-items: center;
      gap: 15px;
    }

    .header a {
      color: white;
      text-decoration: none;
      padding: 8px 16px;
      border: 1px solid rgba(255,255,255,0.3);
      border-radius: 4px;
      transition: background-color 0.3s;
    }

    .header a:hover {
      background-color: rgba(255,255,255,0.1);
    }

    .container {
      max-width: 800px;
      margin: 20px auto;
      background: #fff;
      border-radius: 8px;
      border: 1px solid var(--gray-border);
      overflow: hidden;
    }

    .chat-history {
      max-height: 400px;
      overflow-y: auto;
      padding: 20px;
      border-bottom: 1px solid var(--gray-border);
    }

    .chat-item {
      margin-bottom: 20px;
      padding-bottom: 15px;
      border-bottom: 1px solid #f0f0f0;
    }

    .chat-item:last-child {
      border-bottom: none;
    }

    .question {
      background: var(--light-blue);
      padding: 12px;
      border-radius: 6px;
      margin-bottom: 8px;
      border-left: 4px solid var(--main-blue);
    }

    .answer {
      background: #f8f8f8;
      padding: 12px;
      border-radius: 6px;
      border-left: 4px solid #666;
    }

    .timestamp {
      font-size: 12px;
      color: #666;
      margin-top: 5px;
    }

    .chat-form {
      padding: 20px;
    }

    input[type="text"] {
      width: 100%;
      padding: 14px;
      margin-bottom: 15px;
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

    .error-box {
      padding: 20px;
      margin: 20px;
      border-left: 4px solid #cc0000;
      border-radius: 6px;
      background: #ffe6e6;
      color: #800000;
    }

    .no-chats {
      text-align: center;
      padding: 40px;
      color: #666;
    }

    @media (max-width: 768px) {
      .container {
        margin: 10px;
      }

      .header {
        flex-direction: column;
        gap: 10px;
        text-align: center;
      }

      .header .user-info {
        flex-direction: column;
        gap: 8px;
      }
    }
  </style>
</head>
<body>
  <div class="header">
    <h1>Asistentul tău cu documente 📄</h1>
    <div class="user-info">
      <span>Bună, {{ current_user.username }}!</span>
      <a href="{{ url_for('clear_history') }}" onclick="return confirm('Ești sigur că vrei să ștergi istoricul?')">Șterge istoricul</a>
      <a href="{{ url_for('logout') }}">Ieși din cont</a>
    </div>
  </div>

  <div class="container">
    {% if vector_store_ready %}
      <div class="chat-history">
        {% if chat_history %}
          {% for chat in chat_history %}
            <div class="chat-item">
              <div class="question">
                <strong>Întrebare:</strong> {{ chat.question }}
                <div class="timestamp">{{ chat.timestamp.strftime('%d.%m.%Y %H:%M') }}</div>
              </div>
              <div class="answer">
                <strong>Răspuns:</strong> {{ chat.answer }}
              </div>
            </div>
          {% endfor %}
        {% else %}
          <div class="no-chats">
            Încă nu ai pus nicio întrebare. Începe o conversație mai jos!
          </div>
        {% endif %}
      </div>

      <div class="chat-form">
        <form method="post">
          <input type="text" name="question" placeholder="Pune o întrebare..." required>
          <input type="submit" value="Răspunde">
        </form>
      </div>
    {% else %}
      <div class="error-box">
        <strong>⚠ Documentele nu sunt încărcate.</strong>
        <p>Verifică dosarul documentelor și reîncarcă aplicația.</p>
      </div>
    {% endif %}
  </div>
</body>
</html>
"""


# Routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Numele de utilizator există deja.')
            return render_template_string(LOGIN_TEMPLATE, title='Înregistrare', is_register=True,
                                          submit_text='Înregistrează-te')

        if User.query.filter_by(email=email).first():
            flash('Email-ul este deja folosit.')
            return render_template_string(LOGIN_TEMPLATE, title='Înregistrare', is_register=True,
                                          submit_text='Înregistrează-te')

        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        session['user_id'] = user.id
        return redirect(url_for('index'))

    return render_template_string(LOGIN_TEMPLATE, title='Înregistrare', is_register=True,
                                  submit_text='Înregistrează-te')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username_or_email = request.form['username']
        password = request.form['password']

        # Try to find user by username or email
        user = User.query.filter(
            (User.username == username_or_email) | (User.email == username_or_email)
        ).first()

        if user and user.check_password(password):
            session['user_id'] = user.id
            return redirect(url_for('index'))
        else:
            flash('Date de autentificare incorecte.')

    return render_template_string(LOGIN_TEMPLATE, title='Autentificare', is_register=False, submit_text='Loghează-te')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/clear_history')
@login_required
def clear_history():
    user_id = session['user_id']
    Chat.query.filter_by(user_id=user_id).delete()
    db.session.commit()
    return redirect(url_for('index'))


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    global vector_store, processed_files

    user_id = session['user_id']
    current_user = User.query.get(user_id)

    # Check if refresh is requested
    if request.args.get('refresh') == '1':
        print("Refreshing documents...")
        create_vector_store_from_folder()

    # Initialize vector store if not already done
    if vector_store is None:
        print("Initializing vector store...")
        create_vector_store_from_folder()

    # Get user's chat history
    chat_history = Chat.query.filter_by(user_id=user_id).order_by(Chat.timestamp.desc()).limit(20).all()
    chat_history.reverse()  # Show oldest first

    if request.method == "POST":
        question = request.form.get("question")

        if not question:
            pass  # Handle empty question
        elif vector_store is None:
            pass  # Handle no documents
        else:
            try:
                # Search for relevant documents
                relevant_docs = vector_store.similarity_search(question, k=3)
                context = "\n".join([doc.page_content for doc in relevant_docs])

                if context.strip():
                    # Generate answer
                    answer = generate_answer(context, question)

                    # Save chat to database
                    chat = Chat(user_id=user_id, question=question, answer=answer)
                    db.session.add(chat)
                    db.session.commit()

                    # Redirect to avoid form resubmission
                    return redirect(url_for('index'))

            except Exception as e:
                print(f"Error: {str(e)}")

    # Get updated chat history after potential new message
    chat_history = Chat.query.filter_by(user_id=user_id).order_by(Chat.timestamp.desc()).limit(20).all()
    chat_history.reverse()

    return render_template_string(CHAT_TEMPLATE,
                                  current_user=current_user,
                                  chat_history=chat_history,
                                  vector_store_ready=vector_store is not None)


if __name__ == "__main__":
    print(f"Document folder set to: {DOCUMENTS_FOLDER}")
    print("Creating database tables...")

    with app.app_context():
        db.create_all()

    print("Starting Flask application...")
    app.run(debug=True)