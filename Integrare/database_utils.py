# database_utils.py
import config
import mariadb
import sys

# Conectare globală la MariaDB
try:
    conn = mariadb.connect(
        user="root",
        password=config.PASSWORD,
        host="localhost",
        port=3310,
        database="chatbot"
    )
    cursor = conn.cursor()
    print("[INFO] Conexiune MariaDB stabilită.")
except mariadb.Error as e:
    print(f"[EROARE] Conectare eșuată: {e}")
    sys.exit(1)

def login_user(username, password):
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    return cursor.fetchone() is not None

def register_user(username, password):
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        return False  # utilizator deja există
    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()
    return True
