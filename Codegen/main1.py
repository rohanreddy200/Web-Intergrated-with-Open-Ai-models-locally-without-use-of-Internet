import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import random

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
user_input = "write a code to multiply two numbers a and b in C++"
input_ids = tokenizer.encode(user_input, return_tensors='pt', max_length=256, truncation=True)

# Set generation parameters for accuracy and determinism
max_length = 256  # Maximum length of the generated sequence
num_beams = 10  # Increase the number of beams for better exploration
length_penalty = 0.5  # Lower length penalty to encourage accurate longer sequences
temperature = 0.3  # Lower temperature for high determinism
top_k = 50  # Controls the diversity of the generated text (higher values = more diverse)
no_repeat_ngram_size = 3  # Avoid repetitive trigrams
early_stopping = True  # Enable early stopping

# Seed the random number generator for reproducibility
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        temperature=temperature,
        top_k=top_k,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        num_beams=num_beams,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=early_stopping,
    )

response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Response:", response)
