from flask import Flask, request, render_template, redirect, session, url_for
from auth import login_user, register_user
import requests

app = Flask(__name__)
app.secret_key = 'super-secret-key'  # schimbă cu un secret real

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form['username']
        pw = request.form['password']
        if login_user(user, pw):
            session['user'] = user
            return redirect('/')
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

@app.route('/', methods=['GET', 'POST'])
def prompt():
    if 'user' not in session:
        return redirect('/login')

    answer = None
    if request.method == 'POST':
        question = request.form['question']
        resp = requests.post("http://localhost:5001/generate", json={"prompt": question})
        if resp.ok:
            answer = resp.json()['answer']
        else:
            answer = "Eroare la generarea răspunsului."

    return render_template('prompt.html', user=session['user'], answer=answer)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
