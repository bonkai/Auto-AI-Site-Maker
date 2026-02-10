import ollama
import logging
import sys
import time
import os
import json
import traceback
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----- File Saver Tool Implementation -----

def file_save(path: str, content: str, mode: str = "w") -> str:
    """Save content to a file with safety checks."""
    logging.info(f"Saving file to path: {path}, mode: {mode}")
    
    start_time = time.time()
    try:
        # Ensure the directory exists
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # Write the file
        with open(path, mode) as f:
            f.write(content)
            
        execution_time = time.time() - start_time
        file_size = os.path.getsize(path)
        success_msg = f"File saved successfully to {path} ({file_size} bytes)"
        logging.info(f"{success_msg} (took {execution_time:.2f}s)")
        return success_msg
            
    except Exception as e:
        tb = traceback.format_exc()
        error_msg = f"Error saving file: {str(e)}\n{tb}"
        logging.error(error_msg)
        return error_msg

def file_read(path: str) -> str:
    """Read content from a file with safety checks."""
    logging.info(f"Reading file from path: {path}")
    
    start_time = time.time()
    try:
        if not os.path.exists(path):
            error_msg = f"File not found: {path}"
            logging.error(error_msg)
            return error_msg
            
        with open(path, "r") as f:
            content = f.read()
            
        execution_time = time.time() - start_time
        file_size = os.path.getsize(path)
        success_msg = f"File read successfully from {path} ({file_size} bytes):\n\n{content}"
        logging.info(f"File read successfully from {path} ({file_size} bytes) (took {execution_time:.2f}s)")
        return success_msg
            
    except Exception as e:
        tb = traceback.format_exc()
        error_msg = f"Error reading file: {str(e)}\n{tb}"
        logging.error(error_msg)
        return error_msg

# ----- Tool Registration -----

# File tool definitions
REGISTERED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "file_save",
            "description": "Saves content to a file. Can create new files or overwrite/append to existing ones.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path where content should be saved."
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to save to the file."
                    },
                    "mode": {
                        "type": "string",
                        "description": "File mode: 'w' (write/overwrite), 'a' (append).",
                        "default": "w",
                        "enum": ["w", "a"]
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Reads content from a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to read from."
                    }
                },
                "required": ["path"]
            }
        }
    }
]

# Tool dispatcher map
TOOL_DISPATCHER = {
    "file_save": file_save,
    "file_read": file_read
}

# ----- Model Interaction Functions -----

def send_query(model_name, query):
    """Send a query to the specified model with tool access."""
    system_message = (
        "You are an AI assistant with access to file reading and writing tools. "
        "When the user asks you to save content to a file or read from a file, use the appropriate tool. "
        "Always show your reasoning before using the tool. "
        "Be cautious with operations that might overwrite or delete important files."
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
                            {"role": "system", "content": "You have access to file reading and writing tools."},
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
    """Main function to run the file tool assistant."""
    model_name = "aratan/mistral-small-3.1:24b"
    print(f"=== File Tool Assistant - Using Model: {model_name} ===")
    print("Type 'exit' or 'quit' to end the session\n")
    
    try:
        while True:
            query = input("\nEnter your file operation request: ")
            
            if query.lower() in ['exit', 'quit']:
                print("Exiting file tool assistant.")
                break
                
            response = send_query(model_name, query)
            process_tool_response(response, query)
            
    except KeyboardInterrupt:
        print("\nExiting file tool assistant.")
    except Exception as e:
        print(f"\nError in main loop: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 