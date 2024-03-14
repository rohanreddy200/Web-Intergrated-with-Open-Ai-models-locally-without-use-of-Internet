from flask import Flask, request, jsonify, redirect, url_for
import psycopg2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    dbname='postgres',
    user='postgres',
    password='Saritha8*',
    host='localhost',
    port='5433',
    options='-c search_path=codebot'
)
cursor = conn.cursor()

# Load tokenizer and model
try:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-multi")
    
    print("Tokenizer and model loaded successfully!")
except Exception as e:
    print("Error loading tokenizer or model:", e)
    raise

@app.route('/login', methods=['POST'])
def login():
    try:
        # Get email and password from the login form
        email = request.form['email']
        password = request.form['password']
        
        # Query the database to check if the email and password match
        cursor.execute('SELECT * FROM login_details WHERE email = %s AND password = %s', (email, password))
        user = cursor.fetchone()

        if user:
            # Redirect to the frontend page upon successful login
            return redirect(url_for('frontend'))
        else:
            # Return an error message if login fails
            return jsonify({'error': 'Invalid email or password'}), 401

    except Exception as e:
        return jsonify({'error': f'Server error: {e}'}), 500

@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        # Get user_id and user_input from JSON request
        user_id = request.json.get('user_id')
        user_input = request.json.get('user_input')
        
        # Validate user_id and user_input
        if not user_id or not user_input:
            return jsonify({'error': 'User ID or user input not provided'}), 400

        # Tokenize user input
        input_ids = tokenizer.encode(user_input, return_tensors='pt')

        # Generate response
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1)

        # Decode response
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': f'Server error: {e}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
