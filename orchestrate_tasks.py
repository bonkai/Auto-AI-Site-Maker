#!/usr/bin/env python3
import sys
import subprocess
import json
import logging
import traceback
import ollama  # Import Ollama for dynamic command generation

# Set up logging for this orchestrator script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TERMINAL_SCRIPT_PATH = "./terminal_script.py"  # Path to your terminal_script.py (adjust if needed)
# COMMAND_GENERATION_MODEL = "mistral" # Model for command generation - you can use a smaller, faster model
COMMAND_GENERATION_MODEL = "spratling/mistral-small-3.1-24B-it-2503:Q8_0"  # Model for command generation

def get_decomposed_tasks(complex_task_description: str, decomposition_script_path: str = "./task_decomposition_script.py") -> list[str]:
    """
    Runs the task decomposition script and returns the list of subtasks.
    """
    try:
        process = subprocess.run(
            [decomposition_script_path, complex_task_description],
            capture_output=True,
            text=True,
            check=True # Raise an exception for non-zero return codes
        )
        output_lines = process.stdout.strip().split('\n')
        subtasks = []
        start_subtasks = False
        for line in output_lines:
            if line.startswith("Decomposed Subtasks:"):
                start_subtasks = True
                continue # Skip the header line
            if start_subtasks and line.strip() and line.strip()[0].isdigit(): # Only process lines that are numbered subtasks
                subtasks.append(line.strip())
        return subtasks
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running task decomposition script: {e}")
        logging.error(f"Script output (stderr):\n{e.stderr}")
        return []
    except FileNotFoundError:
        logging.error(f"Task decomposition script not found at: {decomposition_script_path}")
        return []

def generate_terminal_command(subtask_description: str, model_name: str = COMMAND_GENERATION_MODEL) -> str | None:
    """
    Uses an LLM to generate a terminal command for a given subtask description.
    """
    prompt = f"""
You are an AI expert in generating simple, beginner-level terminal commands.
Your goal is to create a single, valid terminal command that directly accomplishes the following subtask:

Subtask Description: "{subtask_description}"

Focus on using basic commands like: mkdir, touch, echo, cat, cp, mv, rm, sed, and very simple combinations of these.  Avoid complex scripting, loops, or external tools beyond standard command-line utilities.

The command should be directly executable in a standard Linux/Unix-like terminal (like bash).

Output ONLY the terminal command itself, with no explanation or extra text.  If you cannot generate a suitable simple terminal command, output nothing.

Example Subtask: Create a new directory named 'project-files'.
Example Command: mkdir project-files

Example Subtask: Add the text "Hello, world!" to a file named 'output.txt'.
Example Command: echo "Hello, world!" >> output.txt

Now, generate a terminal command for the following subtask:
"{subtask_description}"
"""
    try:
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        command = response['message']['content'].strip()
        if not command: # Handle empty command responses
            return None
        return command
    except Exception as e:
        logging.error(f"Error generating command for subtask '{subtask_description}': {e}")
        return None


def execute_subtask_with_terminal_script(subtask_description: str) -> str:
    """
    Executes a single subtask using the terminal_script.py, dynamically generating the command.
    """
    command_to_execute = generate_terminal_command(subtask_description)

    if not command_to_execute:
        logging.warning(f"Could not generate a terminal command for subtask: '{subtask_description}'. Skipping.")
        return f"Warning: No command generated for subtask: '{subtask_description}'. Skipped."

    logging.info(f"Executing subtask: '{subtask_description}' with generated command: '{command_to_execute}'")

    try:
        process = subprocess.run(
            [TERMINAL_SCRIPT_PATH, command_to_execute],
            capture_output=True,
            text=True,
            check=False # Do not raise exception if terminal_script returns non-zero
        )
        output = process.stdout
        error_output = process.stderr
        return_code = process.returncode

        if return_code == 0:
            logging.info(f"Subtask '{subtask_description}' executed successfully.")
            return f"Subtask '{subtask_description}' successful.\nCommand: {command_to_execute}\nOutput:\n{output}\nError (if any):\n{error_output}"
        else:
            logging.error(f"Subtask '{subtask_description}' failed (return code: {return_code}).")
            return f"Subtask '{subtask_description}' failed.\nCommand: {command_to_execute}\nReturn Code: {return_code}\nOutput:\n{output}\nError:\n{error_output}"

    except FileNotFoundError:
        error_msg = f"Error: terminal_script.py not found at: {TERMINAL_SCRIPT_PATH}"
        logging.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error executing subtask '{subtask_description}': {e}"
        logging.error(error_msg)
        logging.error(traceback.format_exc()) # Log detailed traceback for debugging
        return error_msg


def main():
    if len(sys.argv) > 1:
        complex_task = " ".join(sys.argv[1:])
    else:
        complex_task = "Create a basic webpage structure with a heading." # Default task

    print(f"Complex Task: {complex_task}")
    print("==============================")

    subtasks = get_decomposed_tasks(complex_task)

    if not subtasks:
        print("Failed to get decomposed subtasks. Please check task_decomposition_script.py and logs for errors.")
        return

    print("Decomposed Subtasks:")
    for i, subtask in enumerate(subtasks):
        print(f"{i+1}. {subtask}")
    print("\nExecuting subtasks using terminal_script.py and dynamically generated commands...")
    print("==============================")

    for i, subtask in enumerate(subtasks):
        print(f"\n--- Executing Subtask {i+1}/{len(subtasks)}: {subtask} ---")
        result = execute_subtask_with_terminal_script(subtask)
        print(result)
        print("------------------------------")

    print("\n==============================")
    print("Subtask execution completed.")

if __name__ == "__main__":
    main()