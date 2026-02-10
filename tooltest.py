import ollama
import logging
import sys
import time
import os
from io import StringIO
import multiprocessing
import traceback
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----- External Tool Implementation -----

def _run_python_code(code: str, result_dict: dict, safe_globals: dict) -> None:
    """Helper function to run Python code in a separate process."""
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
    """Execute Python code with safety restrictions and timeout."""
    logging.info(f"Executing Python code with timeout {timeout}s")
    logging.info(f"Code to execute:\n{code}")
    
    start_time = time.time()
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
            
            execution_time = time.time() - start_time
            logging.info(f"Execution result: {result['success']} (took {execution_time:.2f}s)")
            return output
    except Exception as e:
        tb = traceback.format_exc()
        error_msg = f"Error in execution environment: {str(e)}\n{tb}"
        logging.error(error_msg)
        return error_msg

# ----- Tool Registration -----

# Simple tool definition
REGISTERED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python_execute",
            "description": "Executes Python code string. Use print statements to see results.",
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

# Tool dispatcher map
TOOL_DISPATCHER = {
    "python_execute": external_python_execute
}

# ----- Model Interaction Functions -----

def send_query(model_name, query):
    """Send a query to the specified model with tool access and time the response."""
    system_message = (
        "You are an AI assistant with access to a Python execution tool. "
        "When the user asks you to perform calculations or run code, use the python_execute tool. "
        "Always show your reasoning before using the tool."
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
                import json
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
                            {"role": "system", "content": "You have access to a Python execution tool."},
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

# ----- Test Scenarios -----

def run_test_scenario(model_name, results_dir):
    """Run a standard test to evaluate how well the model uses the tool."""
    test_queries = [
        "Calculate the sum of all prime numbers between 50 and 100.",
        "Create a Python function to generate the first 10 Fibonacci numbers, then calculate their average.",
        "Generate 100 random numbers between 1 and 1000, then show me their mean, median, and standard deviation.",
        "Write Python code to count how many words in this sentence contain the letter 'e': 'The quick brown fox jumps over the lazy dog and runs through the dense forest quickly'.",
        "Create a simple Python simulation of flipping a coin 1000 times. What percentage of flips came up heads? Show both the code and the result."
    ]
    
    results = {
        "model": model_name,
        "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "queries": [],
        "tool_usage_rate": 0
    }
    
    successful_tool_uses = 0
    print(f"\n===== TESTING MODEL: {model_name} =====\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i}: '{query}' ---\n")
        query_start_time = time.time()
        try:
            response = send_query(model_name, query)
            tool_used = process_tool_response(response, query)
            if tool_used:
                successful_tool_uses += 1
                
            query_total_time = time.time() - query_start_time
            print(f"\nTotal time for query {i}: {query_total_time:.2f}s")
            
            results["queries"].append({
                "query": query,
                "tool_used": tool_used,
                "time": query_total_time
            })
        except Exception as e:
            print(f"Error during test: {str(e)}")
            results["queries"].append({
                "query": query,
                "tool_used": False,
                "error": str(e),
                "time": time.time() - query_start_time
            })
    
    # Calculate tool usage rate
    results["tool_usage_rate"] = (successful_tool_uses / len(test_queries)) * 100
    print(f"\n===== COMPLETED TESTING MODEL: {model_name} =====\n")
    print(f"Tool usage rate: {results['tool_usage_rate']:.1f}% ({successful_tool_uses}/{len(test_queries)} queries)")
    
    # Save results to the specific directory
    import json
    model_safe_name = model_name.replace(':', '_').replace('/', '_')
    filename = os.path.join(results_dir, f"model_test_{model_safe_name}.json")
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")
    
    return results

def test_my_models():
    """Test my specific set of models with standard test queries."""
    my_models = [
        "aratan/mistral-small-3.1:24b",
        "gemma3:latest", 
        "gemma3:27b",
        "deepseek-r1:70b", 
        "deepseek-r1:8b",
        "qwq:latest"
    ]
    
    # Create unique results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"model_comparison_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Saving all results to directory: {results_dir}")
    
    all_results = []
    
    print(f"\n===== STARTING MULTI-MODEL TEST =====")
    print(f"Testing {len(my_models)} models: {', '.join(my_models)}\n")
    
    for model in my_models:
        print(f"\n{'-'*50}")
        print(f"Testing model: {model}")
        print(f"{'-'*50}\n")
        
        try:
            result = run_test_scenario(model, results_dir)
            all_results.append(result)
        except Exception as e:
            print(f"Error testing model {model}: {str(e)}")
            all_results.append({
                "model": model,
                "error": str(e),
                "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tool_usage_rate": 0
            })
        
        print(f"\nCompleted testing {model}\n")
    
    # Display comparison summary
    print("\n===== MODEL COMPARISON SUMMARY =====\n")
    print(f"{'Model':<25} | {'Tool Usage Rate':<15} | {'Avg Response Time':<15}")
    print(f"{'-'*25} | {'-'*15} | {'-'*15}")
    
    for result in all_results:
        model = result.get("model", "unknown")
        usage_rate = result.get("tool_usage_rate", 0)
        
        # Calculate average response time
        query_times = [q.get("time", 0) for q in result.get("queries", [])]
        avg_time = sum(query_times) / len(query_times) if query_times else 0
        
        print(f"{model[:25]:<25} | {usage_rate:>13.1f}% | {avg_time:>13.2f}s")
    
    print("\n===== TEST COMPLETE =====\n")
    
    # Save summary results
    import json
    summary_file = os.path.join(results_dir, "summary_comparison.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Detailed comparison results saved to {summary_file}")
    
    return all_results

if __name__ == "__main__":
    print("LLM Tool Testing Script - Automated Mode")
    print("Testing all specified models automatically with no user interaction required")
    test_my_models()