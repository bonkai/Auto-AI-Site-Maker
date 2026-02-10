#!/usr/bin/env python3
import sys
import time
import logging
import subprocess
import traceback
import json
import ollama
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def create_timestamped_directory(base_dir="projects"):
    """Creates a timestamped directory to avoid overwriting files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir_name = f"project_{timestamp}"
    project_dir_path = os.path.join(base_dir, project_dir_name)
    os.makedirs(project_dir_path, exist_ok=True) # exist_ok to prevent errors if base_dir exists
    logging.info(f"Created project directory: {project_dir_path}")
    return project_dir_path

def change_working_directory(path):
    """Changes the current working directory and logs the change."""
    try:
        os.chdir(path)
        logging.info(f"Changed working directory to: {path}")
        return True
    except OSError as e:
        logging.error(f"Error changing working directory to {path}: {e}")
        return False

# --- Task Decomposition Functions ---
def decompose_task(complex_task_description: str, model_name: str = "mistral") -> list[str]:
    """
    Decomposes a complex task description into a list of very simple, terminal-command-level subtasks.
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
5. **Context Aware:** Assume that all operations should be performed within the current project directory, unless explicitly specified otherwise.
6. **No assumptions about prior knowledge:** Assume the executor starts with a blank slate and needs every detail specified.

Example of good subtask granularity (for creating a file):
1. Create a new directory named 'my_project' inside the current project directory.
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

def rewrite_subtask_as_prompt(subtask_description: str, project_dir: str, model_name: str = "mistral") -> str:
    """
    Rewrites a subtask description into a detailed and simple prompt for a tool call, now project directory aware.
    """
    prompt = f"""
You are an expert prompt engineer, specializing in creating prompts for tools. Your goal is to take a simple subtask and rewrite it into a highly detailed, step-by-step prompt that will maximize the success rate of a tool call designed to execute this subtask.  The prompt should be extremely clear, leave no room for ambiguity, and provide all necessary information for a tool to understand and execute the subtask correctly.  Think about the specific parameters, inputs, and expected outputs a tool would need. Focus on making the prompt as simple and actionable as possible for a tool.

Assume that the current working directory is the project directory: "{project_dir}".  All file and directory operations should be performed relative to this project directory unless explicitly instructed otherwise.

The subtask is:
"{subtask_description}"

Rewrite this subtask into a detailed and simple prompt for a tool.  The rewritten prompt should be:

1. **Extremely Detailed:** Include every necessary detail, parameter, and step required to complete the subtask. Assume the tool is very literal and needs explicit instructions.
2. **Simple and Unambiguous:** Use clear, concise language. Avoid jargon or complex phrasing. The prompt should be easily understood by a tool.
3. **Action-Oriented:** Start with a clear action verb.  Tell the tool *exactly* what to do.
4. **Parameter-Specific:** If the subtask involves parameters or inputs, clearly specify them in the prompt.  For example, if it's about creating a file, specify the filename, path (relative to the project directory), and expected content.
5. **Project Directory Context:** All paths should be relative to the project directory "{project_dir}".  Make sure the tool understands this context.
6. **Focus on Tool Success:** The ultimate goal is to create a prompt that will lead to a successful tool call.  Think about what would make a tool most likely to succeed in the context of the project directory.

Example of a subtask: "Create a file named 'report.txt'"

Good Rewritten Prompt: "Create a new text file named 'report.txt' inside the current project directory "{project_dir}". Ensure the file is empty initially. Use the appropriate tool command to create an empty file in the current directory."

Bad Rewritten Prompt: "Make report.txt" (Too brief and lacks detail)


Now, rewrite the following subtask into a detailed and simple prompt for a tool, keeping in mind the project directory context:
"{subtask_description}"

Please provide just the rewritten prompt.
"""

    try:
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        rewritten_prompt = response['message']['content'].strip()
        return rewritten_prompt
    except Exception as e:
        logging.error(f"Error rewriting subtask as prompt: {e}")
        return f"Error rewriting subtask as prompt: {e}"

# --- Tool Execution Functions ---
def execute_terminal_command(command: str, timeout: int = 30) -> str: # Increased default timeout
    """Execute a terminal command with a timeout, now working in the project directory."""
    logging.info(f"Executing terminal command: '{command}' with timeout {timeout}s in directory: {os.getcwd()}") # Log current directory
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False # Do not raise an exception for non-zero return codes
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        return_code = result.returncode

        log_output = f"Command: '{command}'\nReturn Code: {return_code}"
        if stdout:
            log_output += f"\nSTDOUT:\n{stdout}"
        if stderr:
            log_output += f"\nSTDERR:\n{stderr}"

        if return_code == 0:
            logging.info(f"Command executed successfully:\n{log_output}")
            output_message = f"Command executed successfully.\n{log_output}"
        else:
            logging.warning(f"Command failed with return code {return_code}:\n{log_output}")
            output_message = f"Command failed with return code {return_code}.\n{log_output}"

        return output_message

    except subprocess.TimeoutExpired:
        logging.warning(f"Command timed out after {timeout}s: '{command}'")
        return f"Execution timed out after {timeout} seconds for command: '{command}'"
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"Error executing terminal command: {str(e)} for command: '{command}'\n{tb}")
        return f"Error executing command: '{command}': {str(e)}\n{tb}"

# Register the terminal_execute tool
REGISTERED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "terminal_execute",
            "description": "Executes a terminal command and returns its output. Use this tool to perform actions in the operating system's terminal.  It is crucial for tasks that involve file manipulation, running scripts, or system commands. All commands are executed in the current project directory. Ensure all file paths in commands are relative to the current project directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The terminal command to execute. Be very specific and ensure the command is correct for the operating system. Provide the full command, including all arguments and options. Double-check the syntax. All file paths should be relative to the current project directory."
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Maximum execution time in seconds for the command.  A longer timeout might be needed for complex commands. Default is 30 seconds.",
                        "default": 30
                    }
                },
                "required": ["command"]
            }
        }
    }
]

# Map tool names to their implementations
TOOL_DISPATCHER = {
    "terminal_execute": execute_terminal_command
}

def send_query_with_tool_call(model_name: str, prompt: str) -> dict:
    """
    Send a query to the model with tool registration, specifically designed for tool calls.
    The system prompt is more directive, encouraging tool use.
    """
    messages = [
        {
            "role": "system",
            "content": ("You are a helpful AI assistant designed to execute tasks by using tools when necessary, especially the 'terminal_execute' tool. "
                        "You are operating within a project directory. All file and directory operations should be performed relative to this directory. "
                        "When the user provides a task that can be accomplished via terminal commands, you MUST use the 'terminal_execute' tool to perform the task. "
                        "Carefully analyze the user's prompt to determine the exact terminal command needed. Ensure all file paths are relative to the current project directory. "
                        "If a prompt clearly asks for a terminal action, immediately use the 'terminal_execute' tool with the appropriate command. "
                        "Do not explain how to do it, just do it using the tool. Focus on direct tool invocation based on the prompt, always within the context of the project directory.")
        },
        {"role": "user", "content": prompt}
    ]
    start_time = time.time()
    try:
        response = ollama.chat(
            model=model_name,
            messages=messages,
            tools=REGISTERED_TOOLS
        )
        response["_timing"] = {"initial_response": time.time() - start_time}
        return response
    except Exception as e:
        logging.error(f"Error sending query: {str(e)}")
        return {"error": str(e)}

def process_tool_calls(response: dict) -> list[str]: # Return a list of tool call results
    """
    Process tool calls in the model's response and execute them.
    Returns a list of results from each tool call.
    """
    tool_call_results = []
    message = response.get("message", {})
    tool_calls = message.get("tool_calls", [])

    if not tool_calls:
        logging.info("No tool calls detected in the response.")
        return tool_call_results

    for tool_call in tool_calls:
        function_data = tool_call.get("function", {})
        tool_name = function_data.get("name")
        arguments_str = function_data.get("arguments", "{}")

        logging.debug(f"Arguments string before processing: {arguments_str}, type: {type(arguments_str)}") # Debug logging

        if isinstance(arguments_str, str): # Check if arguments_str is a string
            try:
                arguments = json.loads(arguments_str)
            except json.JSONDecodeError as e:
                error_message = f"Error parsing tool call arguments (JSONDecodeError): {e}. Arguments string was: '{arguments_str}'"
                logging.error(error_message)
                tool_call_results.append(error_message)
                continue # Skip to next tool call or end
        elif isinstance(arguments_str, dict): # If it's already a dict, use it directly
            arguments = arguments_str
        else: # Handle unexpected type
            error_message = f"Unexpected type for tool call arguments: {type(arguments_str)}. Expected str or dict. Arguments were: {arguments_str}"
            logging.error(error_message)
            tool_call_results.append(error_message)
            continue


        logging.info(f"Invoking tool '{tool_name}' with arguments: {arguments}")
        if tool_name in TOOL_DISPATCHER:
            try:
                result = TOOL_DISPATCHER[tool_name](**arguments)
                logging.info(f"Tool '{tool_name}' execution result:\n{result}")
                tool_call_results.append(result)
            except Exception as e:
                error_message = f"Error executing tool '{tool_name}': {e}"
                logging.error(error_message)
                tool_call_results.append(error_message)

        else:
            error_message = f"Tool '{tool_name}' is not recognized."
            logging.error(error_message)
            tool_call_results.append(error_message)

    return tool_call_results

# --- Main Function (Combined Workflow) ---
def main():
    if len(sys.argv) > 1:
        complex_task = " ".join(sys.argv[1:])
    else:
        complex_task = "Create a website about cats with lazers. in a dark futuristic style cyberpunk theme." # Default task

    decomposition_model_name = "gemma3:27b" # Model for task decomposition
    prompt_rewrite_model_name = "gemma3:27b" # Model for rewriting prompts
    tool_execution_model_name = "spratling/mistral-small-3.1-24B-it-2503:Q8_0" # Model for tool execution

    project_dir = create_timestamped_directory() # Create project directory
    if not change_working_directory(project_dir): # Change to project directory
        print(f"Failed to change working directory to {project_dir}. Exiting.")
        return

    print(f"Complex Task: {complex_task}")
    print(f"Project Directory: {project_dir}") # Inform user about project directory
    print("==============================")

    subtasks = decompose_task(complex_task, decomposition_model_name)

    if subtasks:
        print("Decomposed Subtasks:")
        for i, subtask in enumerate(subtasks):
            print(f"{i+1}. {subtask}")

        rewritten_prompts = []
        print("\nRewriting Subtasks into Detailed Prompts:")
        print("---------------------------------------")
        for i, subtask in enumerate(subtasks):
            rewritten_prompt = rewrite_subtask_as_prompt(subtask, project_dir, prompt_rewrite_model_name) # Pass project_dir
            rewritten_prompts.append(rewritten_prompt)
            print(f"\nSubtask: {i+1}. {subtask}")
            print(f"Rewritten Prompt: {rewritten_prompt}")
            print("---------------------------------------")

        print("\nExecuting Rewritten Prompts using Tool Calls:")
        print("===============================================")

        all_tool_results = []
        for i, prompt in enumerate(rewritten_prompts):
            print(f"\n--- Executing Prompt {i+1}/{len(rewritten_prompts)} ---")
            print(f"Prompt: {prompt}")
            response = send_query_with_tool_call(tool_execution_model_name, prompt)

            if "error" in response:
                print(f"Error from model: {response['error']}")
                all_tool_results.append({"prompt": prompt, "error": response['error']})
            else:
                print("\nModel Response:")
                print(response.get("message", {}).get("content", "No content in response."))
                tool_results = process_tool_calls(response)
                all_tool_results.append({"prompt": prompt, "tool_results": tool_results})

            print("------------------------------")

        print("\n--- Summary of Tool Execution ---")
        for result_item in all_tool_results:
            print(f"\nPrompt: {result_item['prompt']}")
            if "error" in result_item:
                print(f"Model Error: {result_item['error']}")
            elif "tool_results" in result_item:
                if not result_item["tool_results"]:
                    print("No tool calls made by the model for this prompt.")
                else:
                    for i, tool_result in enumerate(result_item["tool_results"]):
                        print(f"Tool Call {i+1} Result:\n{tool_result}")
            print("------------------------------")


    else:
        print("Failed to decompose the task.")

if __name__ == "__main__":
    main()