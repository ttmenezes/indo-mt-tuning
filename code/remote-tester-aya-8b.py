from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from peft import PeftModel

# Device configuration
device = "mps"  # Using MPS for Mac

# Configuration parameters
BASE_MODEL_PATH = "cohereforai/aya-expanse-8b"  # Use HF model path directly
ADAPTER_PATH = "../ft-models/aya-expanse-javanese-sft"  # Adapter path

# First load the base model
model = AutoModelForCausalLM.from_pretrained(
          BASE_MODEL_PATH,
          torch_dtype=torch.float16,
          device_map="auto",
        )

# Then load the adapter using PEFT with assign=True
model = PeftModel.from_pretrained(
    model, 
    ADAPTER_PATH,
    is_trainable=False,
    assign=True  # Add this parameter
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

def get_message_format(prompts):
  messages = []

  for p in prompts:
    messages.append(
        [{"role": "user", "content": p}]
      )

  return messages

def generate_aya(
      model,
      prompts,
      temperature=0.75,
      top_p=1.0,
      top_k=0,
      max_new_tokens=1024
    ):

  messages = get_message_format(prompts)

  input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_tensors="pt",
      )
  input_ids = input_ids.to(model.device)
  prompt_padded_len = len(input_ids[0])

  gen_tokens = model.generate(
        input_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        do_sample=True,
      )

  # get only generated tokens
  gen_tokens = [
      gt[prompt_padded_len:] for gt in gen_tokens
    ]

  gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
  return gen_text

# Load test data
def load_translation_pairs():
    # Read the CSV file
    df = pd.read_csv('./test.csv')
    return df

# Get translation pairs
df = load_translation_pairs()
print(f"Loaded {len(df)} translation pairs for testing")

# Prepare all prompts at once
all_prompts = [f'Translate from Javanese to Indonesian: "{text}"' for text in df['javanese']]

# Generate all translations at once
print("Generating translations...")
all_generated = generate_aya(model, all_prompts)

# Create a DataFrame with results
results_df = pd.DataFrame({
    'javanese': df['javanese'],
    'ground_truth': df['indonesian'],
    'generated': all_generated
})

# Save to CSV
results_df.to_csv('translation_results_aya_8b_finetuned.csv', index=False)
print(f"Results saved to translation_results_aya_8b_finetuned.csv")

