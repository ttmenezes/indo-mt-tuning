from transformers import MT5ForConditionalGeneration, MT5Tokenizer, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import torch

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(8) if device.type == "cuda" else None
print(f"Using {'CUDA' if device.type == 'cuda' else 'CPU'}")

# Load model and tokenizer
model = MT5ForConditionalGeneration.from_pretrained(
    # "./models/mt5-base",
    "./outputs/mt5-base-finetuned",
    torch_dtype=torch.bfloat16,
    device_map={"": device}
)
tokenizer = MT5Tokenizer.from_pretrained("./outputs/mt5-base-finetuned")
print(f"Model device: {next(model.parameters()).device}")

# Load and prepare training dataset
file_path = './translation_data_simple_seq2seq.jsonl'
df = pd.read_json(path_or_buf=file_path, lines=True)
dataset = Dataset.from_pandas(df)

# Check for empty targets
empty_targets = [ex for ex in dataset if not ex['target'] or ex['target'].strip() == ""]
print(f"Number of examples with empty targets: {len(empty_targets)}")
dataset = dataset.filter(lambda x: x['target'] and x['target'].strip() != "")

# Load and prepare validation dataset
file_path = './translation_data_simple_seq2seq_validation.jsonl'
val_df = pd.read_json(path_or_buf=file_path, lines=True)
val_dataset = Dataset.from_pandas(val_df)

# Prepare the dataset for MT5 (simpler format, no prompting needed)
def preprocess_function(examples):
    inputs = examples['text']
    targets = examples['target']
    
    # Add padding but with smaller max length
    model_inputs = tokenizer(
        inputs, 
        max_length=64,  # Reduced from 128
        truncation=True, 
        padding="max_length",
        return_tensors="pt"
    )
    
    # Add padding but with smaller max length
    labels = tokenizer(
        targets,
        max_length=64,  # Reduced from 128
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Replace padding token id with -100 for loss calculation
    labels = labels["input_ids"].masked_fill(labels.input_ids == tokenizer.pad_token_id, -100)
    model_inputs["labels"] = labels
    return model_inputs

dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)

val_dataset = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names
)

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=val_dataset,
    args=TrainingArguments(
        per_device_train_batch_size=2,  # Reduced batch size
        gradient_accumulation_steps=16,  # Increased gradient accumulation
        warmup_steps=200,  # Increased warmup
        max_steps=1000,    # Increased steps
        learning_rate=1e-5,  # Reduced learning rate
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=100,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",  # Changed to cosine
        max_grad_norm=0.5,  # Reduced max gradient norm
        seed=3407,
        output_dir="outputs",
        report_to="none",
        bf16=True,
        gradient_checkpointing=True,
    )
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained("outputs/mt5-base-finetuned")
    tokenizer.save_pretrained("outputs/mt5-base-finetuned")
    print("Model and tokenizer saved to local")