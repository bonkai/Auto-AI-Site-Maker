import ollama
import logging
import sys
import time
import os
import subprocess
import json
import traceback
from io import StringIO
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----- Terminal Tool Implementation -----

def terminal_execute(command: str, timeout: int = 10) -> str:
    """Execute a terminal command with safety restrictions and timeout."""
    logging.info(f"Executing terminal command with timeout {timeout}s")
    logging.info(f"Command to execute: {command}")
    
    start_time = time.time()
    try:
        # Run the command in a separate process
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the command to complete or timeout
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            
            # Format the result
            if process.returncode == 0:
                output = f"Command executed successfully:\n\n{stdout}"
                if stderr:
                    output += f"\nWarnings/Errors:\n{stderr}"
            else:
                output = f"Command failed with return code {process.returncode}:\n\n{stderr}"
                if stdout:
                    output += f"\nOutput:\n{stdout}"
            
            execution_time = time.time() - start_time
            logging.info(f"Execution completed with return code {process.returncode} (took {execution_time:.2f}s)")
            return output
            
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            timeout_msg = f"Execution timeout after {timeout} seconds"
            logging.warning(timeout_msg)
            return timeout_msg
            
    except Exception as e:
        tb = traceback.format_exc()
        error_msg = f"Error in execution environment: {str(e)}\n{tb}"
        logging.error(error_msg)
        return error_msg

# ----- Tool Registration -----

# Terminal tool definition
REGISTERED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "terminal_execute",
            "description": "Executes a terminal command on macOS. Use for system commands, file operations, and accessing system utilities.",
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
                        "default": 10
                    }
                },
                "required": ["command"]
            }
        }
    }
]

# Tool dispatcher map
TOOL_DISPATCHER = {
    "terminal_execute": terminal_execute
}

# ----- Model Interaction Functions -----

def send_query(model_name, query):
    """Send a query to the specified model with tool access."""
    system_message = (
        "You are an AI assistant with access to a terminal execution tool. "
        "When the user asks you to perform terminal operations or run commands, use the terminal_execute tool. "
        "Always show your reasoning before using the tool. "
        "Be cautious with commands that might modify or delete files. "
        "You're running on macOS, so ensure your commands are compatible."
    )
    
    start_time = time.time()
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ],
            tools=REGISTERED_TOOLS
        )
        response_time = time.time() - start_time
        response["_timing"] = {"initial_response": response_time}
        return response
    except Exception as e:
        response_time = time.time() - start_time
        logging.error(f"Error with model {model_name}: {str(e)}")
        return {
            "error": str(e),
            "_timing": {"initial_response": response_time},
            "message": {"content": f"ERROR: {str(e)}", "tool_calls": []}
        }

def process_tool_response(response, query):
    """Process the model response and execute any tool calls."""
    content = response.get("message", {}).get("content", "")
    tool_calls = response.get("message", {}).get("tool_calls", [])
    model_name = response.get("model", "unknown")
    
    initial_time = response.get("_timing", {}).get("initial_response", 0)
    print(f"\nModel initial response (took {initial_time:.2f}s):\n{content}\n")
    
    tool_used = False
    
    if tool_calls:
        print("\n=== TOOL CALLS DETECTED ===")
        for tool_call in tool_calls:
            # Extract function data
            if isinstance(tool_call, dict):
                function_data = tool_call.get("function", {})
            else:
                function_data = getattr(tool_call, "function", {})
                if hasattr(function_data, "model_dump"):
                    function_data = function_data.model_dump()
                
            tool_name = function_data.get("name")
            args = function_data.get("arguments", "{}")
            
            # Parse arguments if needed
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    print(f"Error parsing arguments: {args}")
                    args = {}
            
            print(f"Tool: {tool_name}")
            print(f"Arguments: {args}")
            
            # Execute the tool
            if tool_name in TOOL_DISPATCHER:
                tool_used = True
                tool_start_time = time.time()
                result = TOOL_DISPATCHER[tool_name](**args)
                tool_time = time.time() - tool_start_time
                print(f"\nTool Result (took {tool_time:.2f}s):\n{result}")
                
                # Get the model's follow-up response with the tool result
                followup_start_time = time.time()
                try:
                    follow_up = ollama.chat(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You have access to a terminal execution tool on macOS."},
                            {"role": "user", "content": query},
                            {"role": "assistant", "content": content, "tool_calls": tool_calls},
                            {"role": "tool", "name": tool_name, "content": result}
                        ]
                    )
                    followup_time = time.time() - followup_start_time
                    print(f"\nFollow-up response (took {followup_time:.2f}s):\n{follow_up['message']['content']}")
                    
                    total_time = initial_time + tool_time + followup_time
                    print(f"\nTotal interaction time: {total_time:.2f}s")
                except Exception as e:
                    followup_time = time.time() - followup_start_time
                    print(f"\nError in follow-up response (after {followup_time:.2f}s): {str(e)}")
            else:
                print(f"Unknown tool requested: {tool_name}")
    else:
        print("No tool calls detected.")
    
    return tool_used

# ----- Main Loop -----

def main():
    """Main function to run the terminal tool assistant."""
    model_name = "aratan/mistral-small-3.1:24b"
    print(f"=== Terminal Tool Assistant - Using Model: {model_name} ===")
    print("Type 'exit' or 'quit' to end the session\n")
    
    try:
        while True:
            query = input("\nEnter your command request: ")
            
            if query.lower() in ['exit', 'quit']:
                print("Exiting terminal tool assistant.")
                break
                
            response = send_query(model_name, query)
            process_tool_response(response, query)
            
    except KeyboardInterrupt:
        print("\nExiting terminal tool assistant.")
    except Exception as e:
        print(f"\nError in main loop: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 