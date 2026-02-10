import ollama
import logging
import sys
import time
import os
import traceback
from io import StringIO
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----- Python Executor Tool Implementation -----

def python_execute(code: str, timeout: int = 10) -> str:
    """Execute Python code with safety restrictions and timeout."""
    logging.info(f"Executing Python code with timeout {timeout}s")
    logging.info(f"Code to execute: {code}")
    
    start_time = time.time()
    
    # Capture stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    
    # Store the result
    result = ""
    
    try:
        # Set up execution timeout using signal (for Unix-based systems)
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution timed out after {timeout} seconds")
        
        # Set the timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        # Execute the code
        exec_globals = {}
        exec(code, exec_globals)
        
        # Clear the alarm
        signal.alarm(0)
        
        # Capture the output
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()
        
        if stderr:
            result = f"Code executed with warnings/errors:\n\n{stderr}\n\nOutput:\n{stdout}"
        else:
            result = f"Code executed successfully:\n\n{stdout}"
        
    except TimeoutError as e:
        result = str(e)
    except Exception as e:
        tb = traceback.format_exc()
        result = f"Error executing Python code: {str(e)}\n{tb}"
    finally:
        # Reset stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        execution_time = time.time() - start_time
        logging.info(f"Python execution completed (took {execution_time:.2f}s)")
        
    return result

# ----- Tool Registration -----

# Python executor tool definition
REGISTERED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python_execute",
            "description": "Executes Python code. Use for data processing, calculations, and automation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute."
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Maximum execution time in seconds.",
                        "default": 10
                    }
                },
                "required": ["code"]
            }
        }
    }
]

# Tool dispatcher map
TOOL_DISPATCHER = {
    "python_execute": python_execute
}

# ----- Model Interaction Functions -----

def send_query(model_name, query):
    """Send a query to the specified model with tool access."""
    system_message = (
        "You are an AI assistant with access to a Python code execution tool. "
        "When the user asks you to run Python code or perform data processing tasks, use the python_execute tool. "
        "Always show your reasoning before using the tool. "
        "Be cautious with code that might modify or delete files."
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
                            {"role": "system", "content": "You have access to a Python code execution tool."},
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
    """Main function to run the Python executor tool assistant."""
    model_name = "aratan/mistral-small-3.1:24b"
    print(f"=== Python Executor Tool Assistant - Using Model: {model_name} ===")
    print("Type 'exit' or 'quit' to end the session\n")
    
    try:
        while True:
            query = input("\nEnter your Python code request: ")
            
            if query.lower() in ['exit', 'quit']:
                print("Exiting Python executor tool assistant.")
                break
                
            response = send_query(model_name, query)
            process_tool_response(response, query)
            
    except KeyboardInterrupt:
        print("\nExiting Python executor tool assistant.")
    except Exception as e:
        print(f"\nError in main loop: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 