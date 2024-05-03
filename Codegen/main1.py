import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,GPT2Tokenizer, GPT2Model
import logging
import random

# Set up basic logging to suppress INFO-level messages
logging.basicConfig(level=logging.ERROR)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Path to the directory containing the model and tokenizer files
model_tokenizer_directory = r"C:\Users\Public\mistralai"

# Verify and update the paths below based on the actual directory structure
model_path = model_tokenizer_directory  # Update the model path

# Load model and tokenizer
print("Loading model and tokenizer...")

try:
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    print("Tokenizer loaded successfully!")
except Exception as e:
    print("Error loading tokenizer:", e)
    raise

try:
    model = GPT2Model.from_pretrained('gpt2-medium')
    print("Model loaded successfully!")
except Exception as e:
    print("Detailed error loading model:", str(e))
    raise


# Test model inference
user_input = "write a code to print fibonacci series for a number n in C++"
input_ids = tokenizer.encode(user_input, return_tensors='tf').input_ids.cuda()

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

# with torch.no_grad():
#     output_ids = model.generate(
#         input_ids,
#          temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512
#     ).to(device)

response = tokenizer.decode(input_ids)
print("Response:", response)