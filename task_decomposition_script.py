#!/usr/bin/env python3
import sys
import time
import logging
import subprocess
import traceback
import json
import ollama

# Set up logging (you can adjust level as needed)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def decompose_task(complex_task_description: str, model_name: str = "mistral") -> list[str]:
    """
    Decomposes a complex task description into a list of very simple, terminal-command-level subtasks.

    Args:
        complex_task_description: The description of the complex task.
        model_name: The name of the Ollama model to use for decomposition.

    Returns:
        A list of strings, where each string is a simple subtask.
    """

    prompt = f"""
You are an expert task decomposition AI. Your goal is to break down complex tasks into extremely simple, beginner-level subtasks that can be executed via terminal commands.  The subtasks should be so simple that a basic script using terminal commands could accomplish each one.  Think of each subtask as something a novice programmer could do with minimal effort.

The complex task is:
"{complex_task_description}"

Please decompose this task into a numbered list of subtasks. Each subtask should be:

1. **Extremely simple and granular:**  Break down even seemingly simple steps further.  Assume the executor has very limited capabilities and needs explicit, step-by-step instructions.  Aim for maximum simplicity, even if it means many subtasks.
2. **Terminal-command focused:** Think about each subtask as something that could be achieved by a series of basic terminal commands (like `mkdir`, `touch`, `echo`, `cp`, `mv`, `rm`, basic text editors like `nano` or `sed` for simple edits, etc.).  The subtasks themselves don't need to *be* terminal commands, but they should be expressible as such.
3. **Ordered logically:** The subtasks should follow a logical order to achieve the overall complex task.
4. **Self-contained where possible:** Each subtask should be relatively independent, minimizing dependencies on prior subtasks where feasible to simplify execution.
5. **No assumptions about prior knowledge:** Assume the executor starts with a blank slate and needs every detail specified.

Example of good subtask granularity (for creating a file):
1. Create a new directory named 'my_project'.
2. Navigate into the 'my_project' directory using the 'cd' command.
3. Create an empty file named 'index.html' inside the 'my_project' directory.
4. Open the 'index.html' file with a text editor (like nano or echo if just adding initial content).
5. Add the basic HTML structure to 'index.html': `<!DOCTYPE html>\n<html>\n<head><title>My Webpage</title></head>\n<body>\n<h1>Hello, World!</h1>\n</body>\n</html>` and save the file.

Example of bad subtask granularity (too complex):
1. Create a webpage with a header and body.  (This is too high-level)

Now, decompose the complex task: "{complex_task_description}" into extremely simple subtasks. Please provide just the numbered list of subtasks, one subtask per line.
"""

    try:
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        subtask_text = response['message']['content']
        subtasks = [line.strip() for line in subtask_text.strip().split('\n') if line.strip() and line.strip()[0].isdigit()] # Split by lines, remove empty lines and lines not starting with numbers for safety
        return subtasks
    except Exception as e:
        logging.error(f"Error decomposing task: {e}")
        return [f"Error decomposing task: {e}"]

def main():
    if len(sys.argv) > 1:
        complex_task = " ".join(sys.argv[1:])
    else:
        complex_task = "Create a star wars themed website about cats with light sabers and lasers. The website should have a header, a main content area, and a footer. The website should be styled with css and the content should be in the main content area." # Default task

    model_name = "gemma3:27b" # Or your preferred model for decomposition (smaller models are fine for this)

    print(f"Complex Task: {complex_task}")
    print("==============================")

    subtasks = decompose_task(complex_task, model_name)

    if subtasks:
        print("Decomposed Subtasks:")
        for i, subtask in enumerate(subtasks):
            print(f"{subtask}")
        print("\nThese subtasks are now ready to be executed by your terminal script.")
    else:
        print("Failed to decompose the task.")

if __name__ == "__main__":
    main()