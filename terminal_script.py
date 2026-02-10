#!/usr/bin/env python3
import sys
import time
import logging
import subprocess
import traceback
import json
import ollama

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def execute_terminal_command(command: str, timeout: int = 15, max_retries: int = 5, retry_delay: int = 5) -> str: # Default max_retries to 5
    """
    Execute a terminal command with a timeout and retry mechanism.

    Args:
        command: The terminal command to execute.
        timeout: Maximum execution time in seconds for each attempt.
        max_retries: Maximum number of retry attempts if the command fails.
        retry_delay: Delay in seconds between retry attempts.

    Returns:
        The output of the command, or an error message if execution fails after retries.
    """
    retry_count = 0
    while retry_count <= max_retries:
        logging.info(f"Executing terminal command (attempt {retry_count + 1}/{max_retries + 1}): '{command}' with timeout {timeout}s")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            if result.returncode == 0:
                output = f"Command executed successfully.\nSTDOUT:\n{result.stdout}"
                if result.stderr:
                    output += f"\nSTDERR:\n{result.stderr}"
                return output
            else:
                error_message = (f"Command failed with return code {result.returncode}.\n"
                                 f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
                logging.warning(f"Command failed (attempt {retry_count + 1}/{max_retries + 1}): {error_message}")
                if retry_count < max_retries:
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    return f"Command failed after {max_retries + 1} attempts.\n{error_message}"

        except subprocess.TimeoutExpired:
            logging.warning(f"Command timed out after {timeout}s (attempt {retry_count + 1}/{max_retries + 1})")
            if retry_count < max_retries:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                return f"Execution timed out after {timeout} seconds and retries exhausted."
        except Exception as e:
            tb = traceback.format_exc()
            logging.error(f"Error executing terminal command (attempt {retry_count + 1}/{max_retries + 1}): {str(e)}")
            logging.error(tb)
            if retry_count < max_retries:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                return f"Error executing command after {max_retries + 1} attempts: {str(e)}\n{tb}"
        retry_count += 1

    return "Maximum retry attempts reached without success." # Should not reach here normally, but as a safety net


# Register the terminal_execute tool - no changes needed here for retry count
REGISTERED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "terminal_execute",
            "description": "Executes a terminal command and returns its output. Includes retry mechanism.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The terminal command to execute."
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Maximum execution time in seconds per attempt.",
                        "default": 15
                    },
                    "max_retries": {
                        "type": "integer",
                        "description": "Maximum number of retry attempts if command fails.",
                        "default": 5 # Default is now 5 in function def and here in tool def for consistency
                    },
                    "retry_delay": {
                        "type": "integer",
                        "description": "Delay in seconds between retry attempts.",
                        "default": 5
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

def send_simple_query(model_name: str, query: str) -> dict:
    """
    Send a simple query to the provided model with tool registration.
    """
    messages = [
        {
            "role": "system",
            "content": ("You are an AI assistant with access to a terminal command execution tool. "
                        "When given a terminal command in the user prompt, you must invoke the terminal_execute tool. "
                        "The terminal_execute tool has a retry mechanism and will retry up to 5 times if a command fails.") # Updated system message to reflect 5 retries
        },
        {"role": "user", "content": query}
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

def process_response(response: dict) -> None:
    """
    Process the model's response, executing any tool calls found, then print the results.
    """
    message = response.get("message", {})
    content = message.get("content", "")
    tool_calls = message.get("tool_calls", [])

    print("Model response:")
    print(content)
    print("------------------------------")

    if not tool_calls:
        print("No tool calls detected in the response.")
        return

    for tool_call in tool_calls:
        function_data = tool_call.get("function", {})
        tool_name = function_data.get("name")
        args = function_data.get("arguments", "{}")

        # Parse arguments (if provided as a JSON string)
        try:
            if isinstance(args, str):
                args = json.loads(args)
        except Exception as e:
            print(f"Error parsing tool call arguments: {e}")
            args = {}

        print(f"Invoking tool '{tool_name}' with arguments: {args}")
        if tool_name in TOOL_DISPATCHER:
            result = TOOL_DISPATCHER[tool_name](**args)
            print("Tool execution result:")
            print(result)
        else:
            print(f"Tool '{tool_name}' is not recognized.")

def main():
    # Use the command-line argument as the terminal command; default if none provided.
    if len(sys.argv) > 1:
        command_to_run = " ".join(sys.argv[1:])
    else:
        command_to_run = "ls -l /nonexistent_directory"  # Example command that might fail initially

    # Use your best-performing model.
    model_name = "spratling/mistral-small-3.1-24B-it-2503:Q8_0"
    query = f"Please execute the following terminal command using the terminal_execute tool: {command_to_run}"

    print(f"Sending query to model '{model_name}':")
    print(query)
    print("==============================")

    response = send_simple_query(model_name, query)
    process_response(response)

if __name__ == "__main__":
    main()