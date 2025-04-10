from autotrain.params import Seq2SeqParams
from autotrain.project import AutoTrainProject
import os
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, Trainer, TrainerCallback
from datasets import load_dataset

# Create output directories with proper permissions
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

params = Seq2SeqParams(
    data_path="data/",
    train_split="train",
    model="google/mt5-small",
    text_column="javanese", 
    target_column="indonesian",
    lr=5e-6,
    batch_size=2,
    epochs=5,
    max_seq_length=64,
    max_target_length=64,
    max_grad_norm=0.5,
    warmup_ratio=0.1,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    fp16=False,
    # Add output directory
    output_dir=output_dir,
    # Add logging directory
    logging_dir=os.path.join(output_dir, "logs"),
    # Disable certain operations that might cause permission issues
    report_to="none",
    project_name="mt5-small-translation-seq2seq-7",
    username="ttmenezes",
    push_to_hub=True,
    token="",
    # Add early stopping if needed
    early_stopping_patience=3
)

# Ensure the script has write permissions in current directory
try:
    project = AutoTrainProject(
        params=params, 
        backend="local", 
        process=True,
        working_dir=output_dir
    )
    project.create()
except PermissionError as e:
    print(f"Permission error: {e}")
    print("Please ensure you have write permissions in the current directory")
    raise

# When loading the model and optimizer
checkpoint_path = "path/to/your/checkpoint"
checkpoint = torch.load(checkpoint_path)

# Load model state
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Set the starting epoch/step count
start_epoch = checkpoint['epoch']  # or whatever key you used to save the epoch
global_step = checkpoint.get('global_step', 0)  # get global step if it exists

# Manually set starting points
start_epoch = 50  # or whatever number you want to start from
global_step = start_epoch * steps_per_epoch  # if you want to maintain step count

# Load dataset
dataset = load_dataset("json", data_files="translation_data_simple.jsonl", split="train")

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=params.epochs,
    per_device_train_batch_size=params.batch_size,
    gradient_accumulation_steps=params.gradient_accumulation_steps,
    learning_rate=params.lr,
    save_strategy="steps",
    save_steps=200,
    evaluation_strategy="steps",
    eval_steps=200,
    logging_dir=os.path.join(output_dir, "logs"),
    # If resuming from checkpoint, uncomment and set the path:
    # resume_from_checkpoint="path/to/your/checkpoint",
)

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="javanese",
)

# Custom checkpoint callback
class CheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint = {
            'epoch': state.epoch,
            'global_step': state.global_step,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict() if trainer.scheduler else None,
        }
        torch.save(
            checkpoint, 
            f'{args.output_dir}/checkpoint_epoch_{int(state.epoch)}_step_{state.global_step}.pt'
        )

# Add the callback
trainer.add_callback(CheckpointCallback())

# Start training
trainer.train() 