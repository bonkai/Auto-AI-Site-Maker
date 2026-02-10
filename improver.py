import requests
import logging
import os
import time
import re

# LLaMA API configuration
API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwq:latest"
# MODEL_NAME = "gemma3:latest"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ask_question(system_prompt, user_prompt):
    logging.info("Preparing to send a question to the AI model.")

    conversation_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": conversation_history,
        "temperature": 0.6, # Slightly higher temperature for more creative improvements
        "max_tokens": 50000,
        "stream": False
    }

    logging.info("Sending request to the LLaMA API.")
    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        logging.info("Received response from the AI model.")
        return result['message']['content'].strip()
    else:
        logging.error(f"Error in API request: {response.status_code} - {response.text}")
        return None

def improve_website(directory_path, iterations=10):
    filepath = os.path.join(directory_path, "index.html")
    improvement_types = ["content", "visuals", "scripts", "functionality"]

    for i in range(iterations):
        improvement_type = improvement_types[i % len(improvement_types)]
        logging.info(f"Iteration {i+1}/{iterations}: Improving {improvement_type}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                current_html = f.read()
        except FileNotFoundError:
            logging.error(f"Error: index.html not found in directory: {directory_path}")
            print(f"Error: index.html not found in directory: {directory_path}")
            return
        except Exception as e:
            logging.error(f"Error reading index.html: {e}")
            print(f"Error reading index.html: {e}")
            return

        system_prompt = f"""You are an AI website improver. Your task is to incrementally improve an existing single-page website (provided in the user prompt) in each iteration.

This iteration's focus is on: **{improvement_type.upper()}**.

Make a small, targeted improvement to the website's HTML, CSS, and/or JavaScript based on the current improvement type.  Do not rewrite the entire website, only make a small addition or modification. Maintain the existing structure and style as much as possible unless the improvement type specifically requires visual changes.

Here are the improvement types you will cycle through:
- **CONTENT**: Add more relevant text, images, or other media to enhance the website's information or appeal.
- **VISUALS**: Improve the CSS styling to make the website more visually appealing, responsive, or user-friendly. This could include layout adjustments, color scheme changes, typography improvements, or responsiveness enhancements.
- **SCRIPTS**: Add or enhance JavaScript to introduce interactive elements or improve user experience. This could be adding simple interactions, form validation, dynamic content updates, or small functional enhancements.
- **FUNCTIONALITY**: Add a small new functional feature or improve an existing one using HTML, CSS, and JavaScript. This should be a minor enhancement, not a complete overhaul.

Your output should be the complete, updated HTML code for the single-page website, enclosed in a single "HTML" code block.

Do not add explanations or commentary, just the code.

Example of expected output:

```html
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


Current website HTML:

{current_html}"""

        user_prompt = f"Improve the website focusing on {improvement_type}."

        ai_response = ask_question(system_prompt, user_prompt)

        if ai_response:
            html_match = re.search(r"```html\n(.*?)\n```", ai_response, re.DOTALL)
            if html_match:
                improved_html = html_match.group(1).strip()
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(improved_html)
                    logging.info(f"Iteration {i+1}: {improvement_type.capitalize()} improved and saved to {filepath}")
                    print(f"Iteration {i+1}: {improvement_type.capitalize()} improved and saved to {filepath}")
                except Exception as e:
                    logging.error(f"Error saving improved HTML: {e}")
                    print(f"Error saving improved HTML: {e}")
            else:
                logging.error("Could not extract HTML content from AI response.")
                print("Could not extract HTML content from AI response.")
        else:
            logging.error("No response received from the AI for improvement.")
            print("No response received from the AI for improvement.")
        time.sleep(1) # Optional: Add a small delay between iterations to avoid overwhelming the API

def main():
    # directory_path = input("Enter the directory containing index.html to improve: ")
    directory_path = "website_1742268338"
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return

    try:
        iterations = int(input("Enter the number of improvement iterations: "))
        if iterations <= 0:
            print("Error: Number of iterations must be a positive integer.")
            return
    except ValueError:
        print("Error: Invalid number of iterations. Please enter an integer.")
        return

    improve_website(directory_path, iterations)
    print("Website improvement process completed.")

if __name__ == "__main__":
    main()