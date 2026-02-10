import ollama  # pip install ollama
import json
import logging
import os
import traceback
import sys
from io import StringIO
import multiprocessing

# Set up logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----- Conversation History -----
conversation_history = []

# ----- External Tools Implementation -----

def external_save_file(content: str, filename: str, mode: str = "w") -> str:
    """Save content to a file with advanced features.
    
    Args:
        content (str): The content to save to the file.
        filename (str): The path where the file should be saved.
        mode (str, optional): The file opening mode. Default is 'w' for write. Use 'a' for append.
    
    Returns:
        str: A message indicating the result of the operation.
    """
    try:
        # Log the absolute path for debugging
        abs_path = os.path.abspath(filename)
        logging.info(f"Saving file to absolute path: {abs_path}")
        
        # Ensure the directory exists
        directory = os.path.dirname(abs_path)
        if directory and not os.path.exists(directory):
            logging.info(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)

        # Write to the file
        with open(abs_path, mode, encoding="utf-8") as file:
            file.write(content)
        
        # Verify the file exists after writing
        if os.path.exists(abs_path):
            file_size = os.path.getsize(abs_path)
            result = f"Content successfully saved to {abs_path} (size: {file_size} bytes)"
            logging.info(result)
            return result
        else:
            result = f"Warning: File was written but cannot be found at {abs_path}"
            logging.warning(result)
            return result
            
    except Exception as e:
        tb = traceback.format_exc()
        error_msg = f"Error saving file: {str(e)}\n{tb}"
        logging.error(error_msg)
        return error_msg

def _run_python_code(code: str, result_dict: dict, safe_globals: dict) -> None:
    """
    Helper function to run Python code in a separate process.
    
    Args:
        code (str): The Python code to execute
        result_dict (dict): A multiprocessing managed dict to store results
        safe_globals (dict): Safe globals dictionary for code execution
    """
    original_stdout = sys.stdout
    try:
        output_buffer = StringIO()
        sys.stdout = output_buffer
        exec(code, safe_globals, safe_globals)
        result_dict["observation"] = output_buffer.getvalue()
        result_dict["success"] = True
    except Exception as e:
        result_dict["observation"] = str(e)
        result_dict["success"] = False
    finally:
        sys.stdout = original_stdout

def external_python_execute(code: str, timeout: int = 5) -> str:
    """
    Execute Python code with safety restrictions and timeout.
    
    Args:
        code (str): The Python code to execute
        timeout (int, optional): Maximum execution time in seconds. Defaults to 5.
        
    Returns:
        str: The output of the code execution or error message
    """
    logging.info(f"Executing Python code with timeout {timeout}s")
    logging.info(f"Code to execute:\n{code}")
    
    try:
        with multiprocessing.Manager() as manager:
            result = manager.dict({
                "observation": "",
                "success": False
            })
            
            # Create a safe globals dictionary
            if isinstance(__builtins__, dict):
                safe_globals = {"__builtins__": __builtins__}
            else:
                safe_globals = {"__builtins__": __builtins__.__dict__.copy()}
            
            # Run the code in a separate process
            proc = multiprocessing.Process(
                target=_run_python_code,
                args=(code, result, safe_globals)
            )
            
            proc.start()
            proc.join(timeout)
            
            # Handle timeout
            if proc.is_alive():
                proc.terminate()
                proc.join(1)
                timeout_msg = f"Execution timeout after {timeout} seconds"
                logging.warning(timeout_msg)
                return timeout_msg
            
            # Format the result
            if result["success"]:
                output = f"Code executed successfully:\n\n{result['observation']}"
            else:
                output = f"Error during execution:\n\n{result['observation']}"
            
            logging.info(f"Execution result: {result['success']}")
            return output
    except Exception as e:
        tb = traceback.format_exc()
        error_msg = f"Error in execution environment: {str(e)}\n{tb}"
        logging.error(error_msg)
        return error_msg

# You can add more external functions here if needed.

# ----- Tools Registration -----

# Each tool is defined as a dict describing its function signature.
REGISTERED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save_file",
            "description": "Saves given content to a file with the specified filename.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The text content to save."
                    },
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to create."
                    },
                    "mode": {
                        "type": "string",
                        "description": "The file opening mode. Default is 'w' for write. Use 'a' for append.",
                        "enum": ["w", "a"],
                        "default": "w"
                    }
                },
                "required": ["content", "filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "python_execute",
            "description": "Executes Python code string. Note: Only print outputs are visible, function return values are not captured. Use print statements to see results.",
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
                        "default": 5
                    }
                },
                "required": ["code"]
            }
        }
    }
]

# Mapping of tool names to actual external functions.
TOOL_DISPATCHER = {
    "save_file": external_save_file,
    "python_execute": external_python_execute,
    # Add additional tool mappings here.
}

# ----- Model Interaction Functions -----

def send_query(query: str) -> dict:
    """
    Send the query to the model with conversation history and system instruction.
    The system instruction instructs the model to output a tool call when needed.
    """
    system_message = (
        "You are a tool-use enabled AI. When you need to perform an external action, "
        "call the corresponding function with its parameters. Do not include extra commentary in your final answer."
    )
    
    # Create messages array with system message and conversation history
    messages = [{"role": "system", "content": system_message}]
    
    # Add all previous conversation history
    messages.extend(conversation_history)
    
    # Add the current user query
    messages.append({"role": "user", "content": query})
    
    response = ollama.chat(
        model="qwq:latest",
        messages=messages,
        tools=REGISTERED_TOOLS
    )
    return response

def process_response(response: dict, user_query: str):
    """
    Process the API response: If a tool call is detected, dispatch the call.
    Otherwise, print the final answer. Also updates conversation history.
    """
    # Add user's query to history
    conversation_history.append({"role": "user", "content": user_query})
    
    message = response.get("message")
    # Convert the pydantic model to a dict.
    if hasattr(message, "model_dump"):
        message_dict = message.model_dump()
    elif hasattr(message, "dict"):
        message_dict = message.dict()
    else:
        message_dict = message

    logging.info("Message keys: " + ", ".join(message_dict.keys()))
    
    tool_calls = message_dict.get("tool_calls")
    content = message_dict.get("content", "")
    
    # Clean up content by removing <think> tags if present
    if "<think>" in content and "</think>" in content:
        # Extract just the thinking content for logging
        thinking_content = content[content.find("<think>")+7:content.find("</think>")]
        logging.info(f"Model thinking: {thinking_content[:100]}...")  # Log first 100 chars
        
        # Remove the thinking part from the visible response
        clean_content = content.replace(content[content.find("<think>"):content.find("</think>")+8], "").strip()
        if clean_content:
            content = clean_content
        # If no content is left, indicate that the model was just thinking
        else:
            content = "[Model was thinking but didn't provide a final answer. Please ask for a specific output.]"
    
    # Add assistant's response to history (without tool calls details for simplicity)
    conversation_history.append({"role": "assistant", "content": content})
    
    if tool_calls:
        logging.info("Tool calls found: " + str(tool_calls))
        for tool_call in tool_calls:
            # Ensure we have a dictionary.
            if isinstance(tool_call, dict):
                tool_call_dict = tool_call
            elif hasattr(tool_call, "model_dump"):
                tool_call_dict = tool_call.model_dump()
            elif hasattr(tool_call, "dict"):
                tool_call_dict = tool_call.dict()
            else:
                logging.error("Cannot convert tool call to dict.")
                continue

            # Extract nested function data.
            if "function" in tool_call_dict:
                function_data = tool_call_dict["function"]
                tool_name = function_data.get("name")
                args = function_data.get("arguments")
            else:
                tool_name = tool_call_dict.get("name")
                args = tool_call_dict.get("arguments")
            logging.info(f"Detected tool call: {tool_name} with arguments: {args}")
            if tool_name in TOOL_DISPATCHER:
                # Ensure args is a dict.
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception as e:
                        logging.error(f"Error parsing arguments: {e}")
                        args = {}
                # Dispatch the external function.
                result = TOOL_DISPATCHER[tool_name](**args)
                logging.info(f"Tool call result: {result}")
                
                # Add tool result to conversation history
                conversation_history.append({"role": "tool", "name": tool_name, "content": str(result)})
            else:
                logging.warning(f"Unknown tool requested: {tool_name}")
    else:
        logging.info("No tool call detected.")
        print("Model response:", content)

# Add this function to help debug conversation history issues
def debug_conversation_history():
    """Print the current conversation history for debugging."""
    print("\n=== CONVERSATION HISTORY ===")
    for i, msg in enumerate(conversation_history):
        role = msg.get("role", "unknown")
        content_preview = msg.get("content", "")[:50] + "..." if len(msg.get("content", "")) > 50 else msg.get("content", "")
        print(f"{i+1}. [{role}]: {content_preview}")
    print("============================\n")

# ----- Main Interactive Loop -----

def main():
    print("Chat with memory enabled. Each message will build on previous context.")
    print("Type 'clear' to clear conversation history, 'history' to view it, or 'quit' to exit.")
    
    # Generic interactive loop.
    while True:
        query = input("Enter your question: ").strip()
        if query.lower() in {"quit", "exit"}:
            break
        elif query.lower() == "clear":
            conversation_history.clear()
            print("Conversation history cleared.")
            continue
        elif query.lower() == "history":
            debug_conversation_history()
            continue
        
        logging.info("Sending query: " + query)
        response = send_query(query)
        logging.info("Full API Response:\n" + str(response))
        process_response(response, query)
        
        # Display conversation length for user awareness
        print(f"[Conversation memory: {len(conversation_history)//2} turns]")

if __name__ == "__main__":
    main()