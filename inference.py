from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and tokenizer
model_name = "ttmenezes/mt5-large-translation-seq2seq-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate_javanese_to_indonesian(text):
    # Prepare the input
    inputs = tokenizer(text, return_tensors="pt", max_length=64, truncation=True)
    
    # Generate translation
    outputs = model.generate(
        **inputs,
        max_length=64,
        num_beams=4,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    # Decode and return the translation
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# Example usage
javanese_text = "Aku pengin mangan sega goreng"
translation = translate_javanese_to_indonesian(javanese_text)

print(f"Javanese: {javanese_text}")
print(f"Indonesian: {translation}")

# You can try more examples
more_examples = [
    "Aku turu neng omah",
    "Opo sing kok karepake?",
    "Dino iki panas banget"
]

for text in more_examples:
    translation = translate_javanese_to_indonesian(text)
    print(f"\nJavanese: {text}")
    print(f"Indonesian: {translation}") 