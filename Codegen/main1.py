import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Set up basic logging to suppress INFO-level messages
logging.basicConfig(level=logging.ERROR)

# Path to the directory containing the model and tokenizer files
model_tokenizer_directory = r"C:\Users\Public\Codegen-350-mil"

# Verify and update the paths below based on the actual directory structure
model_path = model_tokenizer_directory  # Update the model path

# Load model and tokenizer
print("Loading model and tokenizer...")

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

# Test model inference
user_input = "write a code to add two integers in C++"
input_ids = tokenizer.encode(user_input, return_tensors='pt')

# Set generation parameters
max_length = 128  # Maximum length of the generated sequence
temperature = 0.4  # Controls the randomness of the generated text (higher values = more random)
top_k = 60  # Controls the diversity of the generated text (higher values = more diverse)

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        temperature=temperature,
        top_k=top_k,do_sample=True,
        pad_token_id=tokenizer.eos_token_id,  # Ensure proper padding
    )

response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Response:", response)
