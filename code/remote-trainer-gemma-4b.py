# -*- coding: utf-8 -*-
"""gemma_3_4b_sft.py

# Fine-Tuning Gemma 3-4b using Supervised Fine-Tuning (SFT)

This script demonstrates how to fine-tune the Gemma 3-4b model on custom data
using Parameter-Efficient Fine-Tuning (PEFT) with LoRA. The script handles:
- Loading and quantizing the base model
- Setting up the training configuration
- Training the model with LoRA adapters
- Testing the fine-tuned model
"""

# Install dependencies
# !pip install -U bitsandbytes transformers peft accelerate trl datasets sentencepiece wandb

# Optional for faster, lower memory usage attention
# !pip install flash-attn --no-build-isolation

from transformers import AutoTokenizer, Gemma3ForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
import torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer
import pandas as pd

# GPU/CPU Configuration
USE_GPU = True
device = "cuda:0" if USE_GPU else "cpu"

# Training Configuration
QUANTIZE_4BIT = False
USE_GRAD_CHECKPOINTING = True
TRAIN_BATCH_SIZE = 16
TRAIN_MAX_SEQ_LENGTH = 512
USE_FLASH_ATTENTION = False
GRAD_ACC_STEPS = 2

# Model Configuration
MODEL_PATH = "google/gemma-3-4b-it"

# Load Model with Quantization
quantization_config = None
if QUANTIZE_4BIT:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

attn_implementation = "flash_attention_2" if USE_FLASH_ATTENTION else None

model = Gemma3ForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    attn_implementation=attn_implementation,
    torch_dtype=torch.bfloat16,
    # device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def get_message_format(prompts):
    messages = []
    for p in prompts:
        messages.append([
            {
                "role": "user",
                "content": [{"type": "text", "text": p}]
            }
        ])
    return messages

def generate_gemma(
        model,
        prompts,
        temperature=0.75,
        top_p=1.0,
        top_k=0,
        max_new_tokens=1024
    ):
    messages = get_message_format(prompts)
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)
    
    prompt_len = inputs.shape[-1]
    
    gen_tokens = model.generate(
        inputs,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        do_sample=True,
    )
    
    # Get only generated tokens
    gen_tokens = gen_tokens[:, prompt_len:]
    
    gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    return gen_text

# Test inference on base model
prompts = [
    'Translate from Javanese to Indonesian: "Roti-roti sing disajekne nggarai aku nostalgianan. Kabeh model roti jaman biyen, saka tampilane utawa rasa. Rotine enak lan regane uga mirah."'
]

generations = generate_gemma(model, prompts)

for p, g in zip(prompts, generations):
    print("PROMPT", p, "RESPONSE", g, "\n", sep="\n")

# Dataset Setup
# Load dataset from JSONL file
file_path = './translation_data_simple_seq2seq.jsonl'
df = pd.read_json(path_or_buf=file_path, lines=True)
dataset = Dataset.from_pandas(df)

# Check for empty targets
empty_targets = [ex for ex in dataset if not ex['target'] or ex['target'].strip() == ""]
print(f"Number of examples with empty targets: {len(empty_targets)}")
dataset = dataset.filter(lambda x: x['target'] and x['target'].strip() != "")

# Load validation dataset
file_path = './translation_data_simple_seq2seq_validation.jsonl'
val_df = pd.read_json(path_or_buf=file_path, lines=True)
val_dataset = Dataset.from_pandas(val_df)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['text'])):
        # Format for Gemma chat template
        text = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Translate from Javanese to Indonesian: {example['text'][i]}"}]
                },
                {
                    "role": "model", 
                    "content": [{"type": "text", "text": example['target'][i]}]
                }
            ],
            tokenize=False
        )
        output_texts.append(text)
    return output_texts

# Print example prompt and response
print(f"PROMPT\n{dataset['text'][0]}")
print(f"RESPONSE\n{dataset['target'][0]}")

# SFT Model Training Configuration
training_arguments = TrainingArguments(
    output_dir="results",
    num_train_epochs=20,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC_STEPS,
    gradient_checkpointing=USE_GRAD_CHECKPOINTING,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=10,
    learning_rate=1e-3,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    warmup_ratio=0.05,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="none"
)

# LoRA configuration for Gemma 3
peft_config = LoraConfig(
    lora_alpha=16,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
    formatting_func=formatting_prompts_func
)

# Train the model
trainer.train()

# Save the model to disk
trainer.model.save_pretrained(save_directory='./outputs/gemma-3-4b-javanese-sft')
model.config.use_cache = True
model.eval()

# Test the Fine-Tuned Model
# Load Model and LoRA Adapter
loaded_sft_model = Gemma3ForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    attn_implementation=attn_implementation,
    torch_dtype=torch.bfloat16,
    # device_map="auto"
)
loaded_sft_model.load_adapter("./outputs/gemma-3-4b-javanese-sft")

# Test inference on fine-tuned model
prompts = [
    'Translate from Javanese to Indonesian: "Roti-roti sing disajekne nggarai aku nostalgianan. Kabeh model roti jaman biyen, saka tampilane utawa rasa. Rotine enak lan regane uga mirah."'
]

generations = generate_gemma(loaded_sft_model, prompts)

for p, g in zip(prompts, generations):
    print("PROMPT", p, "RESPONSE", g, "\n", sep="\n")
