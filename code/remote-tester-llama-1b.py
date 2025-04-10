from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import pandas as pd
from tqdm import tqdm

USE_GPU = True
if USE_GPU:
    device = "cuda:0"
else:
    device = "cpu"

device = "mps"

# Configuration parameters
QUANTIZE_4BIT = False
USE_FLASH_ATTENTION = False
# BASE_MODEL_PATH = "./models/llama-1b"  # Base model path
# ADAPTER_PATH = "./outputs/llama-1b-javanese-sft-adapter"  # Adapter path
BASE_MODEL_PATH = "../models/llama-1b"  # Base model path
ADAPTER_PATH = "../models/llama-1b-javanese-sft-adapter"  # Adapter path

# Load Model
quantization_config = None
if QUANTIZE_4BIT:
  quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
      bnb_4bit_compute_dtype=torch.bfloat16,
  )

attn_implementation = None
if USE_FLASH_ATTENTION:
  attn_implementation="flash_attention_2"

# First load the base model
model = AutoModelForCausalLM.from_pretrained(
          BASE_MODEL_PATH,
          quantization_config=quantization_config,
          attn_implementation=attn_implementation,
          torch_dtype=torch.bfloat16,
        )

# Then load the adapter
model.load_adapter(ADAPTER_PATH)
model = model.to(device)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.chat_template = """{% for message in messages %}{% if message['role'] == 'user' %}{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}"""
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS token
tokenizer.padding_side = "right"  # Pad on the right side

def get_message_format(prompts):
    messages = []
    for p in prompts:
        messages.append({
            "role": "user",
            "content": p
        })
    return messages

def generate_llama(
        model,
        prompts,
        temperature=0.75,
        top_p=1.0,
        top_k=0,
        max_new_tokens=1024
    ):

    messages = get_message_format(prompts)
    
    # Create inputs with proper attention mask
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    input_ids = inputs.to(model.device)
    prompt_padded_len = input_ids.shape[1]
    
    # Create attention mask (1 for all tokens since we're not padding the input)
    attention_mask = torch.ones_like(input_ids)
    
    gen_tokens = model.generate(
        input_ids,
        attention_mask=attention_mask,  # Add attention mask
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,  # Explicitly set pad_token_id
    )

    gen_tokens = gen_tokens[:, prompt_padded_len:]  # Fix slicing to work with tensors

    gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    return gen_text

# Load test data
def load_translation_pairs():
    # Read the CSV file
    df = pd.read_csv('../datasets/test.csv')

    # Iterate over the javanese and indonesian columns
    for javanese, indonesian in zip(df['javanese'], df['indonesian']):
        yield {
            'source': javanese,
            'target': indonesian
        }

# Get translation pairs
translation_pairs = list(load_translation_pairs())
print(f"Loaded {len(translation_pairs)} translation pairs for testing")

# Lists to store results
javanese_texts = []
ground_truth_translations = []
generated_translations = []

# Process each test example
for pair in tqdm(translation_pairs):
    source_text = pair['source']      # Javanese text
    reference = pair['target']        # Indonesian text

    # Get the model's translation
    prompt = f'Translate from Javanese to Indonesian: "{source_text}"'
    generated = generate_llama(model, [prompt])[0]

    # Store results
    javanese_texts.append(source_text)
    ground_truth_translations.append(reference)
    generated_translations.append(generated)

# Create a DataFrame
results_df = pd.DataFrame({
    'javanese': javanese_texts,
    'ground_truth': ground_truth_translations,
    'generated': generated_translations
})

# Save to CSV
results_df.to_csv('translation_results_llama_1b_finetuned.csv', index=False)
print(f"Results saved to translation_results_llama_1b_finetuned.csv")
