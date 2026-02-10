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

def execute_terminal_command(command: str, timeout: int = 15) -> str:
    """Execute a terminal command with a timeout."""
    logging.info(f"Executing terminal command: '{command}' with timeout {timeout}s")
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
        else:
            output = (f"Command failed with return code {result.returncode}.\n"
                      f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        return output
    except subprocess.TimeoutExpired:
        logging.warning(f"Command timed out after {timeout}s")
        return f"Execution timed out after {timeout} seconds"
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"Error executing terminal command: {str(e)}")
        return f"Error: {str(e)}\n{tb}"

# Register the terminal_execute tool
REGISTERED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "terminal_execute",
            "description": "Executes a terminal command and returns its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The terminal command to execute."
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Maximum execution time in seconds.",
                        "default": 15
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
                        "When given a terminal command in the user prompt, you must invoke the terminal_execute tool.")
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
        command_to_run = "echo 'Hello, World!'"

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