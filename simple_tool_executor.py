import ollama
import json
import time
from tool_orchestrator import REGISTERED_TOOLS, TOOL_DISPATCHER

def execute_one_task(task_description, model_name="aratan/mistral-small-3.1:24b"):
    """
    Executes a single task using the tool API.
    The model MUST return a valid JSON object that contains a key "tool_calls".
    Each tool call must be an object inside that array with the following structure:
    {
      "function": {
           "name": "terminal_execute" or "python_execute",
           "arguments": "{\"command\": \"<your command>\", \"timeout\": <optional timeout>}"
      }
    }
    Do not include any additional text.
    """
    # Use a strict system prompt that forces the output to be pure JSON as described.
    system_message = (
        "You are a tool executor. For any given task, you MUST respond "
        "with EXACTLY one JSON object containing a single key 'tool_calls' "
        "whose value is an array with one object representing a tool call.\n\n"
        "The JSON response must have the following structure and nothing else:\n"
        "{\n"
        "  \"tool_calls\": [\n"
        "      {\n"
        "         \"function\": {\n"
        "             \"name\": \"terminal_execute\" or \"python_execute\",\n"
        "             \"arguments\": \"<a valid JSON string of arguments>\"\n"
        "         }\n"
        "      }\n"
        "  ]\n"
        "}\n\n"
        "Do NOT include any explanations, markdown formatting, or additional keys. "
        "Your response MUST be valid JSON and must not contain any text outside of it."
    )
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Execute this task using one tool call: {task_description}"}
    ]
    
    print(f"Executing task: {task_description}\n")
    
    try:
        response = ollama.chat(
            model=model_name,
            messages=messages,
            tools=REGISTERED_TOOLS
        )
        
        content = response.get("message", {}).get("content", "")
        print("DEBUG - Raw model response:")
        print(content)
        
        # Try to parse the JSON response
        result_json = json.loads(content)
        tool_calls = result_json.get("tool_calls", [])
        
        print(f"\nParsed tool_calls: {tool_calls}")
        
        if not tool_calls:
            print("No tool_calls found in the JSON response!")
            return {"success": False, "error": "No tool_calls in response"}
        
        # Only consider the first tool call
        tool_call = tool_calls[0]
        function_data = tool_call.get("function", {})
        tool_name = function_data.get("name")
        args = function_data.get("arguments", "{}")
        
        # Attempt to parse the arguments if provided as a JSON string
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception as e:
                print(f"Error parsing arguments: {args}")
                args = {}
        
        print(f"Tool selected: {tool_name}")
        print(f"Arguments: {args}")
        
        if tool_name in TOOL_DISPATCHER:
            start_time = time.time()
            result = TOOL_DISPATCHER[tool_name](**args)
            elapsed = time.time() - start_time
            print(f"\nTool Result (took {elapsed:.2f}s):\n{result}")
            return {"success": True, "result": result}
        else:
            print(f"Unknown tool requested: {tool_name}")
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"success": False, "error": str(e)}

def verify_task(verification_task, model_name="aratan/mistral-small-3.1:24b"):
    """Executes a verification task similarly to execute_one_task and analyzes the result."""
    result = execute_one_task(verification_task, model_name)
    
    if not result["success"]:
        return {"success": False, "details": f"Verification failed: {result.get('error', 'Unknown error')}"}
    
    verification_output = result.get("result", "")
    messages = [
        {"role": "system", "content": ("You are a verification analyzer. "
         "Based solely on the verification output, return a JSON object with keys 'success' (true or false) "
         "and 'details' (a description). Do not include any extra text or markdown formatting.")},
        {"role": "user", "content": f"Verification task: {verification_task}\nVerification output: {verification_output}\n"}
    ]
    
    try:
        response = ollama.chat(model=model_name, messages=messages)
        content = response.get("message", {}).get("content", "")
        # Extract pure JSON if wrapped
        result_verif = json.loads(content)
        return result_verif
    except Exception as e:
        return {"success": False, "details": f"Error parsing verification: {str(e)}"}

if __name__ == "__main__":
    print("=== Simple Tool Executor ===")
    while True:
        task = input("\nEnter a task to execute (or 'exit' to quit): ").strip()
        if task.lower() == 'exit':
            break
        
        exec_result = execute_one_task(task)
        if exec_result["success"]:
            ver_task = input("\nEnter a verification task (or press Enter to skip): ").strip()
            if ver_task:
                ver_result = verify_task(ver_task)
                if ver_result.get("success"):
                    print(f"✅ Verification PASSED: {ver_result.get('details', '')}")
                else:
                    print(f"❌ Verification FAILED: {ver_result.get('details', '')}")
        print("\n" + "-"*50)