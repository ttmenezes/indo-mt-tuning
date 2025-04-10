import json
from typing import List, Dict

def read_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

def format_for_sft(chat_data: List[Dict]) -> List[Dict]:
    formatted_data = []
    
    for conversation in chat_data:
        # Each conversation should have exactly 2 messages (user and assistant)
        if len(conversation) != 2:
            continue
            
        formatted = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful translation assistant that translates text from Javanese to Indonesian. Return only the translated text without any additional explanation."
                },
                {
                    "role": "user",
                    "content": conversation[0]["content"]
                },
                {
                    "role": "assistant", 
                    "content": conversation[1]["content"]
                }
            ]
        }
        formatted_data.append(formatted)
    
    return formatted_data

def write_jsonl(data: List[Dict], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def main():
    input_file = "code/translation_train_formatted_chat.jsonl"
    output_file = "code/translation_train_sft.jsonl"
    
    # Read the input data
    chat_data = read_jsonl(input_file)
    
    # Format for SFT training
    sft_data = format_for_sft(chat_data)
    
    # Write to output file
    write_jsonl(sft_data, output_file)
    
    print(f"Converted {len(sft_data)} conversations to SFT format")
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    main() 