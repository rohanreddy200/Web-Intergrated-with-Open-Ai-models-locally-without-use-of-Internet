import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import random

# Set up basic logging to suppress INFO-level messages
logging.basicConfig(level=logging.ERROR)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Path to the directory containing the model and tokenizer files
model_tokenizer_directory = r"C:\Users\Public\New folder\mistral"

# Ensure the latest versions of bitsandbytes and accelerate are installed
try:
    import bitsandbytes as bnb
    from accelerate import Accelerator
except ImportError as e:
    raise ImportError("Please install accelerate and bitsandbytes: `pip install accelerate` and `pip install -i https://pypi.org/simple/ bitsandbytes`")

# Load model and tokenizer
print("Loading model and tokenizer...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_directory)
    print("Tokenizer loaded successfully!")
except Exception as e:
    print("Error loading tokenizer:", e)
    raise

try:
    # Load the model without explicitly passing quantization_config if it's already present
    model = AutoModelForCausalLM.from_pretrained(
        model_tokenizer_directory,
        low_cpu_mem_usage=True,  # Additional memory optimization
    ).to(device)  # Move model to GPU
    print("Model loaded successfully!")
except Exception as e:
    print("Detailed error loading model:", str(e))
    raise

# Test model inference
user_input = "write a code to print fibonacci series for a number n in C++"
input_ids = tokenizer.encode(user_input, return_tensors='pt').to(device)  # Ensure tensor is on GPU

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
        do_sample=True, 
        max_new_tokens=512,
        num_beams=num_beams,
        length_penalty=length_penalty,
        temperature=temperature,
        top_k=top_k,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=early_stopping  
    )

response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Response:", response)
