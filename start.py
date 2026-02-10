import requests
import logging
import os
import time
import re  # For regular expressions to extract HTML content

# LLaMA API configuration
API_URL = "http://localhost:11434/api/chat"
# MODEL_NAME = "qwq:latest"
MODEL_NAME = "gemma3:latest"
# gemma3:latest      c0494fe00251    3.3 GB    3 days ago
# gemma3:27b         30ddded7fba6    17 GB     4 days ago
# deepseek-r1:70b    0c1615a8ca32    42 GB     8 days ago
# deepseek-r1:8b     28f8fd6cdc67    4.9 GB     8 days ago
# qwq:latest         cc1091b0e276    19 GB     9 days ago

# Set up logging (optional, but good for debugging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to send a question to the LLaMA API
def ask_question(question):
    logging.info("Preparing to send a question to the AI model.")

    # Define the system prompt for creating a single-page website
    system_prompt = """Your task is to create a single-page website contained within a single HTML file. This file should incorporate all HTML structure, CSS styling, and JavaScript functionality.

The HTML file should contain the basic structure and content of the website. Think about fundamental HTML elements to organize content into sections. A typical structure might include:

```html
<!DOCTYPE html>
<html>
<head>
    <title>[Website Title - AI's Choice]</title>
    <style>
        /* CSS styles will go here */
    </style>
</head>
<body>
    <header>
        <!-- Website Header Content -->
    </header>
    <main>
        <!-- Main Website Content -->
    </main>
    <footer>
        <!-- Website Footer Content -->
    </footer>
    <script>
        /* JavaScript code will go here */
    </script>
</body>
</html>


All CSS styling should be included within <style> tags in the <head> section to make the website visually appealing, responsive, and user-friendly. Consider using CSS for layout, typography, colors, and responsiveness using media queries.

All JavaScript code should be included within <script> tags, ideally placed before the closing </body> tag, to implement interactive features and enhance the user experience.

Use your creative freedom to choose a website topic and design. The goal is to create a functional, visually appealing, and responsive single-page website demonstrating good HTML, CSS, and JavaScript practices, all within one HTML file.

The output should be a single fenced code block labeled as "HTML". For example:

<!DOCTYPE html>
<html>
<head>
    <title>...</title>
    <style>...</style>
</head>
<body>
    ...
    <script>...</script>
</body>
</html>
```"""

    # Define the conversation history
    conversation_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    # Define the request payload
    payload = {
        "model": MODEL_NAME,
        "messages": conversation_history,
        "temperature": 0.5,
        "max_tokens": 50000,
        "stream": False
    }

    # Send the request to the LLaMA API
    logging.info("Sending request to the LLaMA API.")
    response = requests.post(API_URL, json=payload)

    # Check for a successful response
    if response.status_code == 200:
        result = response.json()
        logging.info("Received response from the AI model.")
        return result['message']['content'].strip()
    else:
        logging.error(f"Error in API request: {response.status_code} - {response.text}")
        return None

# Main function to execute the script
def main():
    question = "Create a webpage with a child-friendly theme that provides a month-by-month breakdown of what a 4th-grade student in America should learn throughout the school year. The design should be engaging and easy to navigate for children, with fun visuals and clear, simple explanations for each month’s key topics across subjects like math, science, reading, and social studies." # Simple question to trigger website creation
    logging.info(f"Asking the question: {question}")

    answer = ask_question(question)

    if answer:
        logging.info("AI's response received and processing.")

        # Extract HTML content from the answer
        html_match = re.search(r"```html\n(.*?)\n```", answer, re.DOTALL)
        if html_match:
            html_content = html_match.group(1).strip()

            # Create a unique directory based on timestamp
            timestamp = str(int(time.time()))
            directory_name = f"website_{timestamp}"
            os.makedirs(directory_name, exist_ok=True)
            filepath = os.path.join(directory_name, "index.html")

            # Save the HTML content to index.html
            try:
                with open(filepath, 'w', encoding='utf-8') as f:  # Specify encoding for broader character support
                    f.write(html_content)
                logging.info(f"Website saved to: {filepath}")
                print(f"Website saved to: {filepath}") # Also print to console for user feedback
            except Exception as e:
                logging.error(f"Error saving HTML file: {e}")
                print(f"Error saving HTML file: {e}")

        else:
            logging.error("Could not extract HTML content from AI response.")
            print("Could not extract HTML content from AI response.")


    else:
        logging.error("No response received from the AI.")
        print("No response received from the AI.")

# Trigger the main function
if __name__ == "__main__":
    main()