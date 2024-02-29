from flask import Flask, request, jsonify
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Add this line to enable CORS for your Flask app

# Path to the directory containing the model and tokenizer files

try:
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
    print("Tokenizer loaded successfully!")
except Exception as e:
    print("Error loading tokenizer:", e)
    raise

try:
    model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-multi")
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    raise

# Dictionary to store conversation history for each user
conversation_history = {}

@app.route('/generate', methods=['POST'])
def generate_response():
    user_id = request.json.get('user_id')
    if user_id is None:
        return jsonify({'error': 'User ID not provided'}), 400
    
    user_input = request.json.get('user_input')

    # Retrieve conversation history for the user or initialize if not present
    history = conversation_history.setdefault(user_id, [])

    # Concatenate user input with conversation history
    input_text = ' '.join(history + [user_input])

    # Tokenize and generate response
    try:
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
    except Exception as e:
        return jsonify({'error': f'Error tokenizing input: {e}'}), 500

    # Generate response
    try:
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1)
    except Exception as e:
        return jsonify({'error': f'Error generating response: {e}'}), 500

    try:
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        return jsonify({'error': f'Error decoding response: {e}'}), 500

    # Update conversation history for the user
    conversation_history[user_id] = history + [user_input, response]

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
