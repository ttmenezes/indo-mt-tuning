import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM
import pandas as pd
import os

# Device configuration
device = "mps"  # Using mps

# Configuration parameters
# MODEL_PATH = "./models/gemma-3-4b-it"  # Direct path to the model checkpoint
MODEL_PATH = "google/gemma-3-4b-it"

# Load the model using Gemma3ForCausalLM
print("Loading model from checkpoint...")
model = Gemma3ForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,  # Use float16 instead of bfloat16 for MPS
    device_map=None,  # Don't use auto device mapping
    use_safetensors=True
)
model.to(device)
print("moved to device")
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("load tokenizer")

# For loading adapter on MPS devices, we need to work around safetensors issue
adapter_path = "../models/gemma-3-4b-javanese-sft-aug-1"
print(f"Loading adapter from {adapter_path}")

# Move model back to CPU temporarily to load adapter
model.to("cpu")
model.load_adapter(adapter_path)
print("loaded adapter on CPU")

# Move model back to MPS with adapter loaded
model.to(device)
print("moved back to MPS")

def get_message_format(prompts):
    messages_list = []
    for p in prompts:
        # Format messages according to Gemma3 format
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": p}]
            }
        ]
        messages_list.append(messages)
    return messages_list

def generate_gemma(
        model,
        prompts,
        temperature=0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=1024
    ):
    
    results = []
    # Process each prompt individually to properly handle input lengths
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}")
        messages = get_message_format([prompt])[0]  # Get single message format
        
        inputs = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate response
        with torch.no_grad():
            generation = model.generate(
                **inputs,
               # temperature=temperature,
                top_p=top_p,
               # top_k=top_k,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        
        # Extract only the new tokens (the response)
        generation = generation[0][input_len:]
        
        # Decode the response
        decoded = tokenizer.decode(generation, skip_special_tokens=True)
        results.append(decoded)
    
    return results

# Load test data
def load_translation_pairs():
    # Read the CSV file
    df = pd.read_csv('../datasets/test.csv')
    return df

# Get translation pairs
df = load_translation_pairs()
print(f"Loaded {len(df)} translation pairs for testing")

# Prepare all prompts at once
all_prompts = [f'Translate from Javanese to Indonesian: "{text}"' for text in df['javanese']]

# Generate all translations at once
print("Generating translations...")
all_generated = generate_gemma(model, all_prompts)

# Create a DataFrame with results
results_df = pd.DataFrame({
    'javanese': df['javanese'],
    'ground_truth': df['indonesian'],
    'generated': all_generated
})

# Save to CSV
results_df.to_csv('translation_results_gemma_4b_aug_1.csv', index=False)
print(f"Results saved to translation_results_gemma_4b_aug_1.csv")

