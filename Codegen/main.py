from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin
import psycopg2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)
app.config["SECRET_KEY"] = "6d11375e6c34fef23d6398deafa65d7d"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.static_folder = 'static'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

conn = psycopg2.connect(
    user='postgres',
    password='Saritha8*',
    host='localhost',
    port='5432'
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

try:
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
    model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-multi")
    print("Tokenizer and model loaded successfully!")
except Exception as e:
    print("Error loading tokenizer or model:", e)
    raise

class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    cursor = conn.cursor()
    cursor.execute("SELECT id, username FROM codebot.login_details WHERE id = %s", (user_id,))
    user_row = cursor.fetchone()
    cursor.close()
    if user_row:
        return User(id=user_row[0], username=user_row[1])
    return None

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        cursor = conn.cursor()
        cursor.execute("SELECT id, username FROM codebot.login_details WHERE email = %s AND password = %s", (email, password))
        user = cursor.fetchone()
        cursor.close()
        if user:
            user_instance = User(id=user[0], username=user[1])
            login_user(user_instance)
            session['user_id'] = user[0]
            return redirect(url_for('frontend'))
        return "Invalid credentials"
    return render_template('login.html')

@app.route('/get-user-id')
@login_required
def get_user_id():
    return jsonify(user_id=session['user_id'])

@app.route('/frontend')
@login_required
def frontend():
    user_id = session.get('user_id', None)
    return render_template('frontend.html', user_id=user_id)

@app.route('/logout', methods=['POST'])
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/generate', methods=['POST'])
@login_required
def generate_response():
    try:
        user_id = request.json.get('user_id')
        user_input = request.json.get('user_input')
        if not user_id or not user_input:
            return jsonify({'error': 'User ID or user input not provided'}), 400
        input_ids = tokenizer.encode(user_input, return_tensors='pt', max_length=256, truncation=True).to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_length=256,
                num_return_sequences=1,
                temperature=0.3,
                do_sample=True,
                top_k=50,
                num_beams=10,
                length_penalty=0.5,
                no_repeat_ngram_size=3,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("Generated response:", generated_text)
        cursor = conn.cursor()
        try:
            timestamp = datetime.now()
            cursor.execute("INSERT INTO codebot.chat_messages (id, message_text, timestamp) VALUES (%s, %s, %s)", (user_id, f'User: {user_input}', timestamp))
            cursor.execute("INSERT INTO codebot.chat_messages (id, message_text, timestamp) VALUES (%s, %s, %s)", (user_id, f'CodeGPT: {generated_text}', timestamp))
            conn.commit()
        except Exception as db_err:
            conn.rollback()
            print("Database error:", db_err)
            return jsonify({'error': 'Database error'}), 500
        finally:
            cursor.close()
        return jsonify({'response': generated_text})
    except Exception as e:
        return jsonify({'error': f'Server error: {e}'}), 500

@app.route('/chat-history', methods=['GET'])
@login_required
def get_chat_history():
    user_id = request.args.get('user_id')
    date = request.args.get('date')  # Expect date in YYYY-MM-DD format
    cursor = conn.cursor()
    # Ensure the correct SQL query is executed to fetch the desired data
    cursor.execute("SELECT message_text, timestamp FROM codebot.chat_messages WHERE id=%s AND DATE(timestamp)=%s ORDER BY timestamp ASC", (user_id, date))
    chat_history = cursor.fetchall()
    cursor.close()
    # Create a list of dictionaries for each message entry, including timestamp and message text
    formatted_history = [{'message_text': message[0], 'timestamp': message[1].isoformat()} for message in chat_history]
    return jsonify({'chat_history': formatted_history})

@app.route('/get-chat-dates', methods=['GET'])
@login_required
def get_chat_dates():   
    
    user_id = session.get('user_id', None)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT DATE(timestamp) AS chat_date FROM codebot.chat_messages WHERE id=%s ORDER BY chat_date DESC", (user_id,))
    dates = cursor.fetchall()
    cursor.close()
    return jsonify({'chat_dates': [date[0].isoformat() for date in dates]})

if __name__ == '__main__':
    app.run(debug=True)
