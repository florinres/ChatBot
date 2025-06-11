from flask import Flask, request, render_template, redirect, session, jsonify
from database_utils import login_user, register_user
import requests

app = Flask(__name__)
app.secret_key = 'super-secret-key'  # schimbă cu un secret real

@app.route('/')
def index():
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form['username']
        pw = request.form['password']
        if login_user(user, pw):
            session['user'] = user
            return redirect('/prompt')
        else:
            return render_template('login.html', error='Date incorecte.')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user = request.form['username']
        pw = request.form['password']
        if register_user(user, pw):
            return redirect('/login')
        else:
            return render_template('register.html', error='Utilizatorul există deja.')
    return render_template('register.html')

@app.route('/prompt', methods=['GET', 'POST'])
def prompt():
    if 'user' not in session:
        return redirect('/login')

    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        if not question:
            return jsonify({'error': 'Întrebarea este goală.'}), 400
        
        try:
            resp = requests.post("http://localhost:5001/generate", json={"prompt": question})
            if resp.ok:
                answer = resp.json().get('answer', 'Răspuns indisponibil.')
                # If request is AJAX (fetch), return JSON
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'answer': answer})
                else:
                    # fallback: render page with answer
                    return render_template('prompt.html', user=session['user'], answer=answer)
            else:
                error_message = 'Eroare la generarea răspunsului.'
        except Exception as e:
            error_message = f'Eroare server: {e}'

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'error': error_message}), 500
        return render_template('prompt.html', user=session['user'], answer=error_message)

    # GET request: no answer yet
    return render_template('prompt.html', user=session['user'], answer=None)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
