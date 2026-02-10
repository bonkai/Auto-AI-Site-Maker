import time
import json
import tool_orchestrator
import re
from datetime import datetime
def main():
    model_name = "aratan/mistral-small-3.1:24b"
    print("=== Enhanced Task Planner & Checker ===")
    original_task = input("Enter your original task: ").strip()
    
    # STEP 1: Break task into atomic subtasks WITH verification steps
    print("\n=== Planning Phase ===")
    planning_query = (
        f"I need to break down this task into very small, atomic subtasks: \"{original_task}\"\n\n"
        f"For EACH action subtask, also create a verification subtask that checks if the action was successful.\n\n"
        f"Example format:\n"
        f"1. Create directory X\n"
        f"2. Verify directory X exists\n"
        f"3. Copy file Y to location Z\n"
        f"4. Verify file Y exists at location Z\n\n"
        f"Ensure each action is simple and each verification step is clear and specific.\n"
        f"Format your response as a JSON array of objects, where each object has:\n"
        f"- 'type': either 'action' or 'verification'\n"
        f"- 'description': a clear description of the subtask\n"
        f"- 'depends_on': the index of the action this verification checks (only for verification tasks)\n\n"
        f"Only respond with the JSON array."
    )
    
    planning_response = tool_orchestrator.send_query(model_name, planning_query)
    
    try:
        planning_content = planning_response.get("message", {}).get("content", "")
        # Extract JSON if wrapped in markdown or other text
        json_match = re.search(r'```(?:json)?(.*?)```', planning_content, re.DOTALL)
        if json_match:
            planning_content = json_match.group(1).strip()
        
        subtasks = json.loads(planning_content)
        if not isinstance(subtasks, list):
            raise ValueError("Expected a JSON array of subtasks")
    except Exception as e:
        print(f"Error parsing planning response: {e}")
        print("Planning output:", planning_content)
        print("Falling back to creating simple action/verification pairs")
        # Create a basic action and verification pair
        subtasks = [
            {"type": "action", "description": original_task},
            {"type": "verification", "description": f"Verify that: {original_task}", "depends_on": 0}
        ]
    
    # Print the subtasks for visibility
    print(f"\nTask broken down into {len(subtasks)} subtasks:")
    for i, subtask in enumerate(subtasks):
        task_type = subtask.get("type", "unknown")
        prefix = "🔍" if task_type == "verification" else "🔧"
        print(f"{i+1}. {prefix} {task_type.upper()}: {subtask.get('description', '')}")
    
    # STEP 2: Execute the subtasks with verification
    all_subtasks_completed = True
    conversation_history = [{
        "role": "user",
        "content": f"I want to: {original_task}\nI'll be executing this as a series of smaller steps."
    }]
    
    # Group subtasks by action-verification pairs for easier processing
    i = 0
    while i < len(subtasks):
        current_subtask = subtasks[i]
        
        # If this is an action task, find its corresponding verification task
        if current_subtask.get("type") == "action":
            action_index = i
            action_description = current_subtask.get("description", "")
            
            # Look for the verification task that depends on this action
            verification_index = None
            for j, task in enumerate(subtasks):
                if (task.get("type") == "verification" and 
                    task.get("depends_on") == action_index):
                    verification_index = j
                    break
            
            if verification_index is None:
                # If no explicit verification found, look for the next verification task
                if i+1 < len(subtasks) and subtasks[i+1].get("type") == "verification":
                    verification_index = i+1
            
            # Execute the action subtask
            print(f"\n=== Executing Action {action_index+1}/{len(subtasks)} ===")
            print(f"Action: {action_description}")
            
            action_completed = False
            max_action_attempts = 10
            action_attempt = 1
            fallback_attempted = False
            
            while action_attempt <= max_action_attempts and not action_completed:
                print(f"\n--- Action Attempt {action_attempt}/{max_action_attempts} ---")
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                # Execute the action
                action_instruction = (
                    f"Execute the following task using only the terminal_execute tool: {action_description}\n"
                    f"Output ONLY your response in the JSON format delimited by the tokens below.\n"
                    f"Do not output any extra text.\n"
                    f"##JSON_START##\n"
                    f'{{"tool_calls": [{{"function": {{"name": "terminal_execute", "arguments": "{{\\"command\\": \\"YOUR_COMMAND_HERE\\"}}"}}}}]}}\n'
                    f"##JSON_END##\n"
                    f"Replace YOUR_COMMAND_HERE with the terminal command to complete the task."
                )
                
                # Track number of tools used before this action
                tools_before = count_tools_in_history(conversation_history)
                
                # Print debug information about the action
                print(f"Sending action instruction to model...")
                
                action_response = tool_orchestrator.send_query(model_name, action_instruction, conversation_history, reset_history=True)
                
                # Add debugging output to see the raw response
                print("\nDEBUG - Raw model response:")
                tool_calls = action_response.get("message", {}).get("tool_calls", [])
                print(f"Has tool_calls: {bool(tool_calls)}")
                print(f"Number of tool_calls: {len(tool_calls)}")
                if len(tool_calls) > 0:
                    print(f"First tool call: {tool_calls[0]}")
                else:
                    print("No tool_calls found in response")
                
                conversation_history = tool_orchestrator.process_tool_response(
                    action_response, action_instruction, conversation_history
                )
                
                # Another debug check after tool processing
                print(f"After processing, tools in history: {count_tools_in_history(conversation_history)}")
                
                # Count tools after action
                tools_after = count_tools_in_history(conversation_history)
                tools_used = tools_after - tools_before
                
                if tools_used == 0:
                    if not fallback_attempted:
                        print("❌ WARNING: No tools were used. Attempting fallback with a simpler prompt...")
                        fallback_instruction = (
                            "SIMPLE: Execute the following task using ONLY the terminal_execute tool: " + action_description + "\n"
                            "Output ONLY your response in the JSON format delimited by the tokens below.\n"
                            "Do not output any extra text.\n"
                            "##JSON_START##\n"
                            '{"tool_calls": [{"function": {"name": "terminal_execute", "arguments": "{\\"command\\": \\"YOUR_COMMAND_HERE\\"}"}}]}\n'
                            "##JSON_END##\n"
                            "Replace YOUR_COMMAND_HERE with the terminal command to complete the task."
                        )
                        fallback_response = tool_orchestrator.send_query(model_name, fallback_instruction, conversation_history, reset_history=True)
                        print("\nDEBUG - Fallback raw model response:")
                        print(fallback_response.get("message", {}).get("content", ""))
                        conversation_history = tool_orchestrator.process_tool_response(fallback_response, fallback_instruction, conversation_history)
                        tools_after = count_tools_in_history(conversation_history)
                        tools_used = tools_after - tools_before
                        fallback_attempted = True
                        if tools_used > 0:
                            print(f"✓ Fallback used {tools_used} tool(s) during this action")
                        else:
                            print("❌ Fallback did not produce any tool calls.")
                    else:
                        print("❌ WARNING: No tools were used during this action!")
                        conversation_history.append({
                            "role": "system",
                            "content": "CRITICAL ERROR: You failed to use any tools to execute the requested action. Using tools is mandatory."
                        })
                        conversation_history.append({
                            "role": "user",
                            "content": "You didn't use any tools (terminal_execute) to perform the action. "
                                       "Please try again WITH tools - you must actually execute commands, not just describe them."
                        })
                        action_attempt += 1
                        continue
                if tools_used > 0:
                    print(f"✓ Used {tools_used} tool(s) during this action")
                
                # Get the latest assistant response
                last_assistant = next((msg for msg in reversed(conversation_history) 
                                     if msg["role"] == "assistant"), None)
                
                if not last_assistant:
                    print("No assistant response received. Moving to next attempt.")
                    action_attempt += 1
                    continue
                
                # Execute the verification step if available
                if verification_index is not None:
                    verification_task = subtasks[verification_index]
                    verification_description = verification_task.get("description", "")
                    
                    print(f"\n--- Verifying Action (Step {verification_index+1}) ---")
                    print(f"Verification: {verification_description}")
                    
                    # Create specific verification instructions with commands to check
                    verification_instruction = (
                        f"VERIFICATION REQUIRED: Use ONE TOOL CALL to verify: {verification_description}\n\n"
                        f"DO NOT RESPOND WITH TEXT COMMANDS - USE THE TOOL API DIRECTLY.\n\n"
                        f"STEPS:\n"
                        f"1. Choose the appropriate terminal_execute command to check the result\n"
                        f"2. Execute it via the tool API\n"
                        f"3. Evaluate the result\n\n"
                        f"When verification is complete, format your response with this exact structure:\n\n"
                        f"{{\"success\": true/false, \"details\": \"what you checked\", \"commands_used\": [\"commands\"]}}\n\n"
                        f"CRITICAL: Writing commands as text will cause verification to FAIL.\n"
                        f"CRITICAL: You must use the official tool API, not code blocks.\n\n"
                        f"Common verification commands: ls -la, find, stat, cat, grep\n\n"
                        f"Current timestamp if needed: {timestamp}\n\n"
                        f"- 'commands_used': list of commands you used to verify\n\n"
                        f"You MUST run actual verification commands and base your answer ONLY on their results."
                    )
                    
                    # Track number of tools used before verification
                    tools_before_verification = count_tools_in_history(conversation_history)
                    
                    # Print debug information about verification request
                    print(f"Sending verification instruction to model...")
                    
                    verification_response = tool_orchestrator.send_query(model_name, verification_instruction, conversation_history)
                    
                    # Display debugging information
                    print(f"Verification response has tool_calls: {'tool_calls' in verification_response.get('message', {})}")
                    conversation_history = tool_orchestrator.process_tool_response(
                        verification_response, verification_instruction, conversation_history
                    )
                    
                    # Count tools after verification
                    tools_after_verification = count_tools_in_history(conversation_history)
                    verification_tools_used = tools_after_verification - tools_before_verification
                    
                    if verification_tools_used == 0:
                        print("❌ ERROR: Verification did not use any tools to check results!")
                        conversation_history.append({
                            "role": "system", 
                            "content": "CRITICAL ERROR: Verification failed because no tools were used. You MUST use terminal_execute or python_execute to verify."
                        })
                        conversation_history.append({
                            "role": "user",
                            "content": "The verification FAILED because you didn't use any tools to check the results. "
                                       "You must use terminal_execute or python_execute to actually check the system state."
                        })
                        action_completed = False
                    else:
                        print(f"✓ Used {verification_tools_used} tool(s) for verification")
                    
                    # Get the verification result
                    verification_assistant = next((msg for msg in reversed(conversation_history) 
                                                if msg["role"] == "assistant"), None)
                    
                    if verification_assistant:
                        try:
                            # Try to extract JSON results from the verification
                            verification_content = verification_assistant.get("content", "")
                            json_match = re.search(r'```(?:json)?(.*?)```', verification_content, re.DOTALL)
                            if json_match:
                                verification_result_str = json_match.group(1).strip()
                            else:
                                verification_result_str = verification_content
                            
                            verification_result = json.loads(verification_result_str)
                            action_completed = verification_result.get("success", False)
                            
                            if action_completed:
                                print(f"✅ Verification PASSED: {verification_result.get('details', '')}")
                                print(f"Commands used: {verification_result.get('commands_used', [])}")
                            else:
                                print(f"❌ Verification FAILED: {verification_result.get('details', '')}")
                                print(f"Commands used: {verification_result.get('commands_used', [])}")
                                
                                # Add feedback for the next attempt
                                feedback = (
                                    f"The action '{action_description}' failed verification. "
                                    f"Details: {verification_result.get('details', 'Unknown error')}\n\n"
                                    f"Try a different approach to accomplish this action. "
                                    f"Be more thorough and careful in your execution."
                                )
                                conversation_history.append({
                                    "role": "user",
                                    "content": feedback
                                })
                        except Exception as e:
                            print(f"Error parsing verification result: {e}")
                            # If we can't parse the verification result, assume it failed
                            action_completed = False
                            
                            # Add generic feedback
                            conversation_history.append({
                                "role": "user",
                                "content": f"The verification step couldn't be parsed properly. Please try again with the action, and be more explicit about what you're doing."
                            })
                    else:
                        print("No verification response received.")
                        action_completed = False
                else:
                    # If no verification task is available, assume success
                    print("No verification task defined. Assuming action completed successfully.")
                    action_completed = True
                
                action_attempt += 1
                time.sleep(1)
            
            if not action_completed:
                all_subtasks_completed = False
                print(f"⚠️ Failed to complete action {action_index+1} after {max_action_attempts} attempts")
            
            # Skip the verification task in the next iteration if we found one
            if verification_index is not None and verification_index > i:
                i = verification_index + 1
            else:
                i += 1
        else:
            # If it's a verification task without a preceding action, just skip it
            i += 1
    
    # STEP 3: Final assessment
    if all_subtasks_completed:
        print("\n=== Final Result ===")
        print("✅ All subtasks completed successfully!")
        
        # Generate a summary of what was accomplished
        summary_query = (
            f"Provide a summary of how the following task was completed: \"{original_task}\". "
            f"Include what was done in each subtask. Be specific about what files were created, modified, etc."
        )
        summary_response = tool_orchestrator.send_query(model_name, summary_query, conversation_history)
        summary_content = summary_response.get("message", {}).get("content", "")
        print("\n=== Task Summary ===")
        print(summary_content)
    else:
        print("\n=== Final Result ===")
        print("⚠️ Not all subtasks could be completed successfully.")
        print("Please review the output and consider breaking down the task differently or providing more specific instructions.")

def count_tools_in_history(conversation_history):
    """Count the number of tool messages in the conversation history"""
    return sum(1 for msg in conversation_history if msg.get("role") == "tool")

if __name__ == "__main__":
    main() 