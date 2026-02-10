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
    system_prompt = "Your task is to create a website consisting of three separate files: index.html, styles.css, and script.js, based on the given specifications.\n\nThe index.html file should contain the basic structure and content of the website. Think about fundamental HTML elements to organize content into sections. For example, a typical structure might include:\n\n```html\n<!DOCTYPE html>\n<html>\n<head>\n    <title>[Website Title - AI's Choice]</title>\n    <link rel=\"stylesheet\" href=\"styles.css\">\n</head>\n<body>\n    <header>\n        <!-- Website Header Content (e.g., navigation, logo) -->\n    </header>\n    <main>\n        <!-- Main Website Content -->\n    </main>\n    <footer>\n        <!-- Website Footer Content (e.g., copyright, links) -->\n    </footer>\n    <script src=\"script.js\"></script>\n</body>\n</html>\n```\n\nThe styles.css file should handle all the styling to make the website visually appealing, responsive across different screen sizes, and user-friendly. Consider using CSS for layout, typography, colors, and responsiveness using media queries. Example CSS concepts to think about:\n\n```css\n/* styles.css - Example CSS Concepts */\nbody {\n    font-family: sans-serif; /* Example: Choose a readable font */\n    margin: 0; /* Example: Reset default margins */\n}\n\n/* Example: Style the header, main content, and footer */\nheader { /* ... styles for header ... */ }\nmain { /* ... styles for main content ... */ }\nfooter { /* ... styles for footer ... */ }\n\n/* Example: Media query for responsiveness (adjust styles for smaller screens) */\n@media (max-width: 768px) {\n    /* Styles to apply when screen width is 768px or less */\n}\n```\n\nThe script.js file should implement interactive features to enhance the user experience. Think about adding simple interactions like:\n\n```javascript\n// script.js - Example Interactive Concepts\ndocument.addEventListener('DOMContentLoaded', () => {\n    // Example: Add interactivity after the page has loaded\n    // You could add event listeners to buttons, create dynamic content, etc.\n});\n```\n\n**Use your creative freedom to choose a website topic and design.**  The goal is to create a functional, visually appealing, and responsive website demonstrating good HTML, CSS, and JavaScript practices. It could be a website about anything you can imagine – a blog, a portfolio, a simple tool, a landing page, etc.\n\nPlease ensure that the HTML, CSS, and JavaScript code in their respective files are well-structured, efficiently organized, and properly commented for readability and maintainability.\n\nThe output should be provided in three separate fenced code blocks, clearly labeled with their respective file extensions: \"HTML\", \"CSS\", and \"JavaScript\". For example:\n\n```html\n```\n\n```css\n/* styles.css content here */\n```\n\n```javascript\n// script.js content here */\n```\n```"
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