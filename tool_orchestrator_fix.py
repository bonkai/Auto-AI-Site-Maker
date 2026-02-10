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
    
    # Debug mode: Check what's in the tool_calls
    print(f"DEBUG: Tool calls type: {type(tool_calls)}")
    print(f"DEBUG: Tool calls found: {len(tool_calls)}")
    
    # Handle text-based tool calls in markdown codeblocks if no real tool calls were found
    if not tool_calls:
        import re
        print("No formal tool calls found. Checking for text-based tool calls in markdown...")
        tool_call_patterns = [
            r'terminal_execute\(command\s*=\s*"([^"]+)"\)',
            r"terminal_execute\(command\s*=\s*'([^']+)'\)",
            r'python_execute\(code\s*=\s*"([^"]+)"\)',
            r"python_execute\(code\s*=\s*'([^']+)'\)"
        ]
        
        for pattern in tool_call_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                print(f"Found potential text-based tool call: {match}")
                # Here you could generate synthetic tool calls and add them to tool_calls
                # This would require adapting your tool execution code
    
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
            
            # Execute the tool
            if tool_name in TOOL_DISPATCHER:
                tool_start_time = time.time()
                result = TOOL_DISPATCHER[tool_name](**args)
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