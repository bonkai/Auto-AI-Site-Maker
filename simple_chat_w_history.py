import os
import json
import datetime
import requests
import logging

# LLaMA API configuration
API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwq:latest"
# MODEL_NAME = "gemma3:latest"
# gemma3:latest      c0494fe00251    3.3 GB    3 days ago    
# gemma3:27b         30ddded7fba6    17 GB     4 days ago    
# deepseek-r1:70b    0c1615a8ca32    42 GB     8 days ago    
# deepseek-r1:8b     28f8fd6cdc67    4.9 GB    8 days ago    
# qwq:latest         cc1091b0e276    19 GB     9 days ago    

# Directory for conversation history
CONVERSATION_DIR = "conversation_history"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_conversation(history):
    """
    Save the conversation history to a JSON file in a new directory.
    The file is named with a timestamp for unique record keeping.
    """
    if not os.path.exists(CONVERSATION_DIR):
        os.makedirs(CONVERSATION_DIR)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(CONVERSATION_DIR, f"conversation_{timestamp}.json")
    with open(filename, "w") as f:
        json.dump(history, f, indent=4)
    logging.info(f"Conversation saved to {filename}")

def ask_question(question, conversation_history):
    """
    Append the user question to the conversation history,
    send it to the LLaMA API with streaming enabled,
    and stream the AI's response token by token while keeping proper spacing.
    """
    conversation_history.append({"role": "user", "content": question})
    
    payload = {
        "model": MODEL_NAME,
        "messages": conversation_history,
        "temperature": 0.5,
        "max_tokens": 50000,
        "stream": True  # Streaming enabled
    }
    
    logging.info("Sending request to the LLaMA API.")
    response = requests.post(API_URL, json=payload, stream=True)

    if response.status_code == 200:
        logging.info("Streaming response from the AI model.")
        answer = ""
        print("\nAssistant:", end=" ", flush=True)  # Print assistant label
        
        last_char = " "  # Track last printed character to maintain spacing
        
        for line in response.iter_lines():
            if line:
                try:
                    token_data = json.loads(line)
                    token = token_data.get("message", {}).get("content", "")

                    if token:
                        # Add a space before printing if the last char wasn't a space
                        if not last_char.isspace() and not token.startswith((".", ",", "!", "?", ":", ";")):
                            print(" ", end="", flush=True)
                            answer += " "
                        
                        print(token, end="", flush=True)  # Stream token properly
                        answer += token
                        last_char = token[-1]  # Update last character
                        
                except json.JSONDecodeError:
                    continue  # Ignore invalid JSON lines
        
        print("\n")  # Newline after full response
        conversation_history.append({"role": "assistant", "content": answer})
        return answer
    else:
        logging.error(f"Error in API request: {response.status_code} - {response.text}")
        return None

def main():
    system_prompt = "You are a hyper-intelligent AI from the future."
    conversation_history = [{"role": "system", "content": system_prompt}]
    
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            logging.info("Exiting conversation.")
            break
        
        answer = ask_question(question, conversation_history)
        
        if answer:
            save_conversation(conversation_history)
        else:
            print("Error: No response received from the AI.")

if __name__ == "__main__":
    main()