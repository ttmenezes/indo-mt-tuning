import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_text(prompt, model, tokenizer, max_length=100):
    # Encode the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Load your model and tokenizer from checkpoint
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Specify your checkpoint path
checkpoint_path = "path/to/your/checkpoint"  # Replace with your checkpoint path

# Load model from checkpoint
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    offload_folder="offload",
)

# Load tokenizer - use the same base model as your checkpoint
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Test the model with some example prompts from your validation set
print("Running test generations...")

# Optional: Load validation data if available
try:
    import pandas as pd
    valid_data = pd.read_csv("datasets/valid.csv")
    test_prompts = valid_data['prompt'].tolist()[:3]  # Test first 3 prompts
except:
    test_prompts = [
        "Write a short poem about artificial intelligence:",
        "Explain what is machine learning in one sentence:",
        "Complete this sentence: The future of AI is..."
    ]

for prompt in test_prompts:
    print("\nPrompt:", prompt)
    response = generate_text(prompt, model, tokenizer)
    print("Response:", response)
    print("-" * 50)

# Memory cleanup
del model
torch.cuda.empty_cache()

# Interactive testing
while True:
    user_input = input("\nEnter your prompt (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    
    response = generate_text(user_input, model, tokenizer)
    print("\nResponse:", response) 