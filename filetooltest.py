import ollama
import logging
import sys
import time
import os
import json
import shutil
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----- External Tool Implementation -----

def external_save_file(content: str, filename: str, mode: str = "w") -> str:
    """Save content to a file.
    
    Args:
        content (str): The content to save to the file.
        filename (str): The path where the file should be saved.
        mode (str, optional): The file opening mode. Default is 'w' for write. Use 'a' for append.
    
    Returns:
        str: A message indicating the result of the operation.
    """
    try:
        # Get absolute path
        abs_path = os.path.abspath(filename)
        logging.info(f"Saving file to: {abs_path}")
        
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
            result = f"File saved successfully: {abs_path} (size: {file_size} bytes)"
            logging.info(result)
            return result
        else:
            result = f"Warning: File was written but cannot be found at {abs_path}"
            logging.warning(result)
            return result
            
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        error_msg = f"Error saving file: {str(e)}\n{tb}"
        logging.error(error_msg)
        return error_msg

def external_read_file(filename: str) -> str:
    """Read content from a file.
    
    Args:
        filename (str): The path of the file to read.
    
    Returns:
        str: The content of the file or an error message.
    """
    try:
        abs_path = os.path.abspath(filename)
        logging.info(f"Reading file from: {abs_path}")
        
        if not os.path.exists(abs_path):
            error_msg = f"Error: File does not exist at {abs_path}"
            logging.error(error_msg)
            return error_msg
            
        with open(abs_path, 'r', encoding="utf-8") as file:
            content = file.read()
            
        file_size = os.path.getsize(abs_path)
        result = f"File read successfully: {abs_path} (size: {file_size} bytes)\n\nContent:\n{content}"
        logging.info(f"Successfully read file: {abs_path} (size: {file_size} bytes)")
        return result
            
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        error_msg = f"Error reading file: {str(e)}\n{tb}"
        logging.error(error_msg)
        return error_msg

# ----- Tool Registration -----

# Define the tools
REGISTERED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save_file",
            "description": "Saves given content to a file with the specified filename. Use this to create or update files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to save in the file."
                    },
                    "filename": {
                        "type": "string",
                        "description": "The path where the file should be saved (e.g., 'website/index.html')."
                    },
                    "mode": {
                        "type": "string",
                        "description": "The file opening mode. 'w' for write (overwrite), 'a' for append.",
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
            "name": "read_file",
            "description": "Reads content from a file at the specified path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The path of the file to read (e.g., 'website/index.html')."
                    }
                },
                "required": ["filename"]
            }
        }
    }
]

# Tool dispatcher map
TOOL_DISPATCHER = {
    "save_file": external_save_file,
    "read_file": external_read_file
}

# ----- Model Interaction Functions -----

def send_query(model_name, query):
    """Send a query to the specified model with tool access and time the response."""
    system_message = (
        "You are an AI assistant specialized in web development with access to tools for creating and modifying files. "
        "When asked to create or update files like HTML, CSS, JavaScript, or other code files, use the save_file tool. "
        "When you need to see the current content of a file before updating it, use the read_file tool. "
        "Always provide clean, well-formatted, and functional code. "
        "When creating a website, focus on making the components work well together."
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

def process_tool_response(response, query, context=None):
    """Process the model response and execute any tool calls."""
    content = response.get("message", {}).get("content", "")
    tool_calls = response.get("message", {}).get("tool_calls", [])
    model_name = response.get("model", "unknown")
    
    initial_time = response.get("_timing", {}).get("initial_response", 0)
    print(f"\nModel initial response (took {initial_time:.2f}s):\n{content}\n")
    
    if context is None:
        context = {"tool_used": False, "files_created": [], "files_updated": []}
    
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
                context["tool_used"] = True
                tool_start_time = time.time()
                
                # Track files created or updated
                if tool_name == "save_file":
                    filename = args.get("filename", "")
                    mode = args.get("mode", "w")
                    if mode == "w" and (not os.path.exists(filename) or os.path.getsize(filename) == 0):
                        context["files_created"].append(filename)
                    else:
                        context["files_updated"].append(filename)
                
                result = TOOL_DISPATCHER[tool_name](**args)
                tool_time = time.time() - tool_start_time
                print(f"\nTool Result (took {tool_time:.2f}s):\n{result}")
                
                # Get the model's follow-up response with the tool result
                followup_start_time = time.time()
                try:
                    follow_up = ollama.chat(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You have access to tools for creating and modifying files."},
                            {"role": "user", "content": query},
                            {"role": "assistant", "content": content, "tool_calls": tool_calls},
                            {"role": "tool", "name": tool_name, "content": result}
                        ]
                    )
                    followup_time = time.time() - followup_start_time
                    
                    # Check if there are more tool calls in the follow-up
                    follow_content = follow_up.get("message", {}).get("content", "")
                    follow_tool_calls = follow_up.get("message", {}).get("tool_calls", [])
                    
                    print(f"\nFollow-up response (took {followup_time:.2f}s):\n{follow_content}")
                    
                    if follow_tool_calls:
                        print("\nFollow-up contains additional tool calls. Processing...")
                        # Recursively process additional tool calls
                        context = process_tool_response(follow_up, query, context)
                    
                except Exception as e:
                    followup_time = time.time() - followup_start_time
                    print(f"\nError in follow-up response (after {followup_time:.2f}s): {str(e)}")
            else:
                print(f"Unknown tool requested: {tool_name}")
    else:
        print("No tool calls detected.")
    
    return context

# ----- Test Scenarios -----

def run_test_scenario(model_name, base_dir):
    """Run a standard test to evaluate how well the model uses file creation tools."""
    
    # Create a unique project directory for this model
    model_safe_name = model_name.replace(':', '_').replace('/', '_')
    project_dir = os.path.join(base_dir, model_safe_name)
    
    # Clean up any existing directory
    if os.path.exists(project_dir):
        shutil.rmtree(project_dir)
    os.makedirs(project_dir, exist_ok=True)
    
    # Save the original directory
    original_dir = os.getcwd()
    
    try:
        # Change to the project directory for relative paths to work correctly
        os.chdir(project_dir)
        
        test_tasks = [
            # Creation tasks - Simplified
            {
                "name": "Create HTML",
                "query": "Create a simple HTML5 page for a company called 'TechInnovate'. Include a header, a main section with some content, and a footer. Save the file as 'index.html'."
            },
            {
                "name": "Create CSS",
                "query": "Create a basic CSS file to style the TechInnovate website. Style the header, main section, and footer with different background colors and proper spacing. Save the file as 'styles.css'."
            },
            {
                "name": "Create JavaScript",
                "query": "Create a very simple JavaScript file that just adds a 'Hello World' alert when the page loads. Save it as 'script.js'."
            },
            {
                "name": "Create Python File",
                "query": "Create a simple Python file that prints 'Hello World' and the current date. Save it as 'hello.py'."
            },
            # Update tasks - Simplified  
            {
                "name": "Update HTML",
                "query": "Read the index.html file and then update it to add a navigation menu in the header with links to Home, About, and Contact pages."
            },
            {
                "name": "Update CSS",
                "query": "Read the styles.css file and then update it to add styling for the navigation menu, making it horizontal with some padding between items."
            }
        ]
        
        results = {
            "model": model_name,
            "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tasks": [],
            "tool_usage_rate": 0,
            "files_created": [],
            "files_updated": []
        }
        
        successful_tool_uses = 0
        print(f"\n===== TESTING MODEL: {model_name} =====\n")
        print(f"Project directory: {project_dir}")
        
        for i, task in enumerate(test_tasks, 1):
            print(f"\n--- Test Task {i}: '{task['name']}' ---\n")
            query_start_time = time.time()
            
            try:
                response = send_query(model_name, task['query'])
                context = process_tool_response(response, task['query'])
                
                if context.get("tool_used", False):
                    successful_tool_uses += 1
                    
                query_total_time = time.time() - query_start_time
                print(f"\nTotal time for task {i}: {query_total_time:.2f}s")
                
                # Add created/updated files to results
                results["files_created"].extend(context.get("files_created", []))
                results["files_updated"].extend(context.get("files_updated", []))
                
                task_result = {
                    "name": task["name"],
                    "tool_used": context.get("tool_used", False),
                    "files_created": context.get("files_created", []),
                    "files_updated": context.get("files_updated", []),
                    "time": query_total_time
                }
                results["tasks"].append(task_result)
                
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"Error during test: {str(e)}\n{tb}")
                results["tasks"].append({
                    "name": task["name"],
                    "tool_used": False,
                    "error": str(e),
                    "time": time.time() - query_start_time
                })
        
        # Calculate tool usage rate
        results["tool_usage_rate"] = (successful_tool_uses / len(test_tasks)) * 100 if test_tasks else 0
        print(f"\n===== COMPLETED TESTING MODEL: {model_name} =====\n")
        print(f"Tool usage rate: {results['tool_usage_rate']:.1f}% ({successful_tool_uses}/{len(test_tasks)} tasks)")
        print(f"Files created: {len(set(results['files_created']))}")
        print(f"Files updated: {len(set(results['files_updated']))}")
        
        # Save results to the project directory
        with open(os.path.join(project_dir, "test_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {os.path.join(project_dir, 'test_results.json')}")
        
        return results
        
    finally:
        # CRITICAL: Always return to the original directory, even if errors occur
        os.chdir(original_dir)

def test_my_models():
    """Test my specific set of models with file creation tasks."""
    my_models = [
        "aratan/mistral-small-3.1:24b",
        "gemma3:latest", 
        "gemma3:27b",
        "deepseek-r1:70b", 
        "deepseek-r1:8b",
        "qwq:latest"
    ]
    
    # Get absolute path for results directory to avoid confusion
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.abspath(f"file_tool_test_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Saving all results to directory: {results_dir}")
    
    all_results = []
    original_directory = os.getcwd()
    
    print(f"\n===== STARTING MULTI-MODEL FILE TOOL TEST =====")
    print(f"Testing {len(my_models)} models: {', '.join(my_models)}\n")
    
    for model in my_models:
        print(f"\n{'-'*50}")
        print(f"Testing model: {model}")
        print(f"{'-'*50}\n")
        
        try:
            # Make sure we're in the original directory for each new model test
            os.chdir(original_directory)
            
            result = run_test_scenario(model, results_dir)
            all_results.append(result)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"Error testing model {model}: {str(e)}\n{tb}")
            all_results.append({
                "model": model,
                "error": str(e),
                "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tool_usage_rate": 0,
                "files_created": [],
                "files_updated": [],
                "tasks": []
            })
        
        # Extra verification that we're back in the original directory
        if os.getcwd() != original_directory:
            print(f"WARNING: Directory changed unexpectedly. Resetting to original directory.")
            os.chdir(original_directory)
            
        print(f"\nCompleted testing {model}\n")
    
    # Display comparison summary
    print("\n===== MODEL COMPARISON SUMMARY =====\n")
    print(f"{'Model':<25} | {'Tool Usage':<10} | {'Files Created':<13} | {'Files Updated':<13} | {'Avg Time':<10}")
    print(f"{'-'*25} | {'-'*10} | {'-'*13} | {'-'*13} | {'-'*10}")
    
    for result in all_results:
        model = result.get("model", "unknown")
        usage_rate = result.get("tool_usage_rate", 0)
        
        # Count unique files
        files_created = len(set(result.get("files_created", [])))
        files_updated = len(set(result.get("files_updated", [])))
        
        # Calculate average response time
        task_times = [t.get("time", 0) for t in result.get("tasks", [])]
        avg_time = sum(task_times) / len(task_times) if task_times else 0
        
        print(f"{model[:25]:<25} | {usage_rate:>8.1f}% | {files_created:>13} | {files_updated:>13} | {avg_time:>8.2f}s")
    
    print("\n===== TEST COMPLETE =====\n")
    
    # Save summary results
    summary_file = os.path.join(results_dir, "summary_comparison.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Detailed comparison results saved to {summary_file}")
    
    return all_results

if __name__ == "__main__":
    print("LLM File Tool Testing Script - Automated Mode")
    print("Testing all specified models for their ability to create and modify files")
    test_my_models() 