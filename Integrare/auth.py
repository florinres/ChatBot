import pymysql

def get_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='parola_ta',
        database='chatbot_db'
    )

def login_user(username, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
    result = cur.fetchone()
    conn.close()
    return result is not None

def register_user(username, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=%s", (username,))
    if cur.fetchone():
        conn.close()
        return False  # user exists
    cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
    conn.commit()
    conn.close()
    return True
