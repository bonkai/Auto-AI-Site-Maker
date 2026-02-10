import ollama
import logging
import sys
import time
import os
import json
import traceback
from datetime import datetime
import importlib
import re

# Import tools modules
from terminal_tool import terminal_execute

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----- Tool Registration -----

# Aggregate all tools
REGISTERED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "terminal_execute",
            "description": "Executes a terminal command on macOS. Use for system operations, file management, and accessing system utilities.",
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

# ----- Orchestrator Model Interaction Functions -----

def send_query(model_name, query, conversation_history=None, reset_history=False):
    """Send a query to the specified model with tool access, maintaining conversation history."""
    if reset_history or conversation_history is None:
        conversation_history = []
        
    system_message = (
        "You are a task executor. Given a task, output ONLY a JSON object with a key \"tool_calls\" "
        "that exactly matches the specified format. Do not output any extra text."
    )
    
    messages = [{"role": "system", "content": system_message}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": query})
    
    start_time = time.time()
    try:
        response = ollama.chat(
            model=model_name,
            messages=messages,
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

def process_tool_response(response, query, conversation_history):
    """Process the model response and execute any tool calls, updating conversation history."""
    content = response.get("message", {}).get("content", "")
    tool_calls = response.get("message", {}).get("tool_calls", [])
    model_name = response.get("model", "unknown")
    
    initial_time = response.get("_timing", {}).get("initial_response", 0)
    print(f"\nOrchestrator planning (took {initial_time:.2f}s):\n{content}\n")
    
    # Add assistant message to conversation history
    conversation_history.append({
        "role": "assistant", 
        "content": content,
        "tool_calls": tool_calls if tool_calls else None
    })
    
    # Fallback: try to search inside a JSON code block if no tool_calls key exists
    if not tool_calls:
        # First try to extract JSON using our marker tokens
        marker_match = re.search(r"##JSON_START##(.*?)##JSON_END##", content, re.DOTALL)
        if marker_match:
            try:
                parsed_json = json.loads(marker_match.group(1).strip())
                tool_calls = parsed_json.get("tool_calls", [])
                response["message"]["tool_calls"] = tool_calls
            except Exception as e:
                logging.error(f"Fallback JSON parse error with markers: {str(e)}")
        else:
            # Fallback to previous approach if markers not found
            json_match = re.search(r"```json(.*?)```", content, re.DOTALL)
            if json_match:
                try:
                    parsed_json = json.loads(json_match.group(1).strip())
                    tool_calls = parsed_json.get("tool_calls", [])
                    response["message"]["tool_calls"] = tool_calls
                except Exception as e:
                    logging.error(f"Fallback JSON parse error with code block: {str(e)}")
    
    if tool_calls:
        print("\n=== EXECUTING TOOL CALLS ===")
        for i, tool_call in enumerate(tool_calls):
            # Extract function data
            if isinstance(tool_call, dict):
                function_data = tool_call.get("function", {})
                tool_call_id = tool_call.get("id", f"call_{i}")
            else:
                function_data = getattr(tool_call, "function", {})
                if hasattr(function_data, "model_dump"):
                    function_data = function_data.model_dump()
                tool_call_id = getattr(tool_call, "id", f"call_{i}")
                
            tool_name = function_data.get("name")
            args = function_data.get("arguments", "{}")
            
            # Parse arguments if needed
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    print(f"Error parsing arguments: {args}")
                    args = {}
            
            print(f"\nTool {i+1}/{len(tool_calls)}: {tool_name}")
            print(f"Arguments: {args}")
            
            # Debug the arguments more thoroughly
            print(f"DEBUG - Raw argument data type: {type(args)}")
            print(f"DEBUG - Raw argument content: {args}")
            
            # Ensure command is present for terminal_execute
            if tool_name == "terminal_execute" and "command" not in args:
                # Check if the entire string was passed as JSON instead of parsing
                if isinstance(args, str) and "command" in args:
                    try:
                        # Try to parse it one more time
                        args = json.loads(args)
                    except:
                        # If still can't parse, extract command using regex as last resort
                        command_match = re.search(r'"command"\s*:\s*"([^"]+)"', args)
                        if command_match:
                            args = {"command": command_match.group(1)}
                        else:
                            # If we still have no command, use args as the command itself
                            args = {"command": f"echo 'Failed to parse command. Raw: {args}'"}
                else:
                    # If no command found, add a default error message
                    print(f"WARNING: No 'command' found in arguments. Using fallback.")
                    args = {"command": f"echo 'Missing command argument in tool call'"}
            
            # Execute the tool
            if tool_name in TOOL_DISPATCHER:
                tool_start_time = time.time()
                try:
                    # Extra safety check for required arguments
                    if tool_name == "terminal_execute" and "command" not in args:
                        result = "ERROR: Missing required 'command' argument"
                    else:
                        result = TOOL_DISPATCHER[tool_name](**args)
                except Exception as e:
                    print(f"ERROR executing tool: {str(e)}")
                    result = f"Tool execution error: {str(e)}"
                tool_time = time.time() - tool_start_time
                print(f"\nTool Result (took {tool_time:.2f}s):\n{result}")
                
                # Add tool result to conversation history
                conversation_history.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": result,
                    "tool_call_id": tool_call_id
                })
            else:
                print(f"Unknown tool requested: {tool_name}")
                conversation_history.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": f"Error: Unknown tool '{tool_name}'",
                    "tool_call_id": tool_call_id
                })
        
        # Get the model's follow-up response after all tools have been executed
        followup_start_time = time.time()
        try:
            follow_up = ollama.chat(
                model=model_name,
                messages=conversation_history
            )
            followup_time = time.time() - followup_start_time
            follow_up_content = follow_up['message']['content']
            print(f"\nTask summary (took {followup_time:.2f}s):\n{follow_up_content}")
            
            # Add the follow-up to conversation history
            conversation_history.append({
                "role": "assistant",
                "content": follow_up_content
            })
            
            # Check if there are more tool calls in the follow-up
            follow_up_tool_calls = follow_up['message'].get('tool_calls', [])
            if follow_up_tool_calls:
                print("\nNeed to execute additional tools to complete the task...")
                process_tool_response(follow_up, "", conversation_history)
                
        except Exception as e:
            followup_time = time.time() - followup_start_time
            error_message = f"\nError in task summary (after {followup_time:.2f}s): {str(e)}"
            print(error_message)
            conversation_history.append({
                "role": "assistant",
                "content": error_message
            })
    else:
        print("No tools needed for this task.")
    
    return conversation_history

# ----- Main Loop -----

def main():
    """Main function to run the tool orchestrator."""
    model_name = "aratan/mistral-small-3.1:24b"
    print(f"=== Tool Orchestrator - Using Model: {model_name} ===")
    print("Available tool: terminal_execute")
    print("Type 'exit' or 'quit' to end the session\n")
    
    conversation_history = []
    
    try:
        while True:
            query = input("\nEnter your task: ")
            
            if query.lower() in ['exit', 'quit']:
                print("Exiting tool orchestrator.")
                break
                
            if query.lower() in ['clear', 'reset']:
                conversation_history = []
                print("Conversation history cleared.")
                continue
                
            response = send_query(model_name, query, conversation_history)
            conversation_history = process_tool_response(response, query, conversation_history)
            
    except KeyboardInterrupt:
        print("\nExiting tool orchestrator.")
    except Exception as e:
        print(f"\nError in main loop: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 