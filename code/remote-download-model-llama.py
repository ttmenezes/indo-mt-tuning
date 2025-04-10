import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

def download_model(model_name: str, output_dir: str):
    """
    Download and save a model and its tokenizer from Hugging Face Hub.
    
    Args:
        model_name: Name of the model on HuggingFace Hub
        output_dir: Local directory to save the model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {model_name} to {output_dir}")
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    try:
        # Download model
        print("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": device}
        )
        
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Save both locally
        print("Saving model and tokenizer locally...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"Successfully downloaded and saved to {output_dir}")
        
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        raise

if __name__ == "__main__":
    # Login to Hugging Face
    login(token="")
    
    # Configuration
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    OUTPUT_DIR = "./models/llama-1b"
    
    # Download the model
    download_model(MODEL_NAME, OUTPUT_DIR)