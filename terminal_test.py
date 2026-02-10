import ollama
import logging
import sys
import time
import os
import json
import subprocess
import traceback
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----- External Tool Implementation -----

def execute_terminal_command(command: str, timeout: int = 15) -> str:
    """Execute a terminal command with timeout."""
    logging.info(f"Executing terminal command with timeout {timeout}s")
    logging.info(f"Command to execute: {command}")
    
    start_time = time.time()
    try:
        # Run the command in a subprocess
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        execution_time = time.time() - start_time
        logging.info(f"Command execution completed in {execution_time:.2f}s")
        
        # Format the output
        if result.returncode == 0:
            output = f"Command executed successfully (return code: {result.returncode})\n\nSTDOUT:\n{result.stdout}"
            if result.stderr:
                output += f"\n\nSTDERR:\n{result.stderr}"
        else:
            output = f"Command failed (return code: {result.returncode})\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        
        return output
    except subprocess.TimeoutExpired:
        logging.warning(f"Command execution timed out after {timeout}s")
        return f"Execution timeout after {timeout} seconds"
    except Exception as e:
        tb = traceback.format_exc()
        error_msg = f"Error executing command: {str(e)}\n{tb}"
        logging.error(error_msg)
        return error_msg

# ----- Tool Registration -----

# Simple tool definition
REGISTERED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "terminal_execute",
            "description": "Executes a terminal command and returns the output.",
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
                        "default": 15
                    }
                },
                "required": ["command"]
            }
        }
    }
]

# Tool dispatcher map
TOOL_DISPATCHER = {
    "terminal_execute": execute_terminal_command
}

# ----- Model Interaction Functions -----

def send_query(model_name, query, force_tool_usage=False, message_history=None):
    """Send a query to the specified model with tool access and time the response."""
    system_message = (
        "You are an AI assistant with access to a terminal command execution tool. "
        "When the user asks you to perform tasks that require terminal commands, use the terminal_execute tool. "
        "Always show your reasoning before using the tool. "
        "Be careful with commands that could modify the system and always explain what your commands do."
    )
    
    if force_tool_usage:
        system_message += (
            "\n\nIMPORTANT: You MUST use the terminal_execute tool to solve the user's request. "
            "Do not provide explanations without using the tool. "
            "This is a test of your ability to use terminal commands effectively."
        )
    
    messages = [{"role": "system", "content": system_message}]
    
    # Add message history if provided
    if message_history:
        messages.extend(message_history)
    
    # Add current query
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

def process_tool_response(response, query, attempt=1, max_attempts=5):
    """Process the model response and execute any tool calls."""
    content = response.get("message", {}).get("content", "")
    tool_calls = response.get("message", {}).get("tool_calls", [])
    model_name = response.get("model", "unknown")
    
    initial_time = response.get("_timing", {}).get("initial_response", 0)
    print(f"\nModel initial response (attempt {attempt}/{max_attempts}, took {initial_time:.2f}s):\n{content}\n")
    
    tool_used = False
    execution_success = False
    tool_content = ""
    
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
                tool_content = result  # Store the result for success evaluation
                tool_time = time.time() - tool_start_time
                print(f"\nTool Result (took {tool_time:.2f}s):\n{result}")
                
                # Simple heuristic to check if the execution was successful
                execution_success = "failed" not in result.lower()[:100] and "error" not in result.lower()[:100]
                
                # Get the model's follow-up response with the tool result
                followup_start_time = time.time()
                try:
                    follow_up = ollama.chat(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You have access to a terminal command execution tool."},
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
    
    return tool_used, execution_success, tool_content

# ----- Test Scenarios -----

def run_test_scenario(model_name, results_dir, max_attempts=5, force_tool_usage=True):
    """Run a progressive test with multiple attempts per task."""
    test_queries = [
        # Basic commands
        "What is the current date and time?",
        "Show me the contents of the current directory.",
        "How much disk space is available on this system?",
        
        # Intermediate commands
        "Find all text files in the current directory and its subdirectories, then count how many there are.",
        "Show me the 10 most recently modified files in this directory.",
        "Check if Python is installed on this system and show its version.",
        
        # Advanced commands
        "Find all files larger than 1MB in the current directory and its subdirectories, sort them by size, and show the top 5.",
        "Show me the number of running processes, grouped by user, sorted by count.",
        "Write a one-liner to find all files containing the word 'error' in the current directory and show their names with line numbers.",
        
        # Complex multi-step commands
        "Create a temporary file with 100 random numbers, calculate their sum, average, and standard deviation using terminal commands.",
        "Get the IP address of this machine, then use it to create a simple HTTP request to check if port 80 is open.",
        
        # Very complex tasks
        "Write a one-liner to find the top 10 most frequent words in all text files in the current directory, ignoring case and common words like 'the', 'a', and 'an'.",
        "Check system performance by showing CPU usage, memory usage, and disk I/O in a formatted table, updating every 5 seconds for 3 iterations.",
        "Create a simple monitoring script that checks if a specified process is running, and if not, logs the time it wasn't found. Run this check 3 times with a 2-second interval.",
        
        # ----- New Shell Script Tests -----
        "Create a simple shell script that prints 'Hello, World!' and run it.",
        "Write a shell script that takes a command line argument and prints it back.",
        "Create a shell script that lists all files in the current directory with their sizes.",
        "Write a shell script that counts the number of lines in all text files in the current directory.",
        "Create a shell script that backs up all .txt files to a new directory with timestamp in the name.",
        "Write a shell script that checks if a process is running and starts it if it's not.",
        "Create a shell script that monitors CPU usage and sends an alert if it exceeds 80%.",
        "Write a shell script that finds all files modified in the last 24 hours and compresses them.",
        "Create a shell script that parses Apache access logs and reports the top 5 IP addresses by hit count.",
        "Write a shell script that implements a simple web server using netcat that serves files from a directory.",
        "Create a shell script that automatically detects and fixes common system issues (low disk space, high CPU usage).",
        "Write a shell script that extracts data from a CSV file, performs calculations, and generates a new CSV with results.",
        "Create a shell script that implements a simple distributed task manager across multiple machines using SSH.",
        "Write a shell script that monitors system resources, collects metrics, and generates a performance report with graphs.",
        "Create a shell script that implements a simple REST API client with authentication and JSON parsing using curl and jq."
    ]
    
    # Update the complexity_labels list to include the new tests
    complexity_labels = [
        "Basic", "Basic", "Basic",
        "Intermediate", "Intermediate", "Intermediate",
        "Advanced", "Advanced", "Advanced",
        "Complex Multi-step", "Complex Multi-step",
        "Very Complex", "Very Complex", "Very Complex",
        # ----- New Shell Script Complexity Labels -----
        "Beginner Shell Script", "Beginner Shell Script", "Beginner Shell Script",
        "Intermediate Shell Script", "Intermediate Shell Script", "Intermediate Shell Script",
        "Advanced Shell Script", "Advanced Shell Script", "Advanced Shell Script",
        "Very Advanced Shell Script", "Very Advanced Shell Script", "Very Advanced Shell Script",
        "Super Advanced Shell Script", "Super Advanced Shell Script", "Super Advanced Shell Script"
    ]
    
    results = {
        "model": model_name,
        "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "queries": [],
        "attempts_per_task": max_attempts,
        "force_tool_usage": force_tool_usage,
        "tool_usage_rate": 0,
        "success_rate": 0,
        "complexity_results": {}
    }
    
    successful_tool_uses = 0
    successful_executions = 0
    print(f"\n===== TESTING MODEL: {model_name} =====\n")
    print(f"Max attempts per task: {max_attempts}")
    print(f"Force tool usage: {force_tool_usage}")
    
    for i, query in enumerate(test_queries):
        complexity = complexity_labels[i]
        print(f"\n--- Test Query {i+1}/{len(test_queries)} [{complexity}]: '{query}' ---\n")
        
        query_result = {
            "query": query,
            "complexity": complexity,
            "attempts": [],
            "best_attempt": None,
            "success": False
        }
        
        message_history = []
        
        for attempt in range(1, max_attempts + 1):
            print(f"\n--- Attempt {attempt}/{max_attempts} ---")
            
            attempt_start_time = time.time()
            
            # On second and subsequent attempts, add a prompt to use the tool if it wasn't used before
            if attempt > 1 and not any(a.get("tool_used", False) for a in query_result["attempts"]):
                prompt = f"Try again. You must use the terminal_execute tool to solve this task: {query}"
            else:
                prompt = query
            
            try:
                response = send_query(model_name, prompt, force_tool_usage, message_history)
                tool_used, exec_success, tool_content = process_tool_response(
                    response, query, attempt=attempt, max_attempts=max_attempts
                )
                
                attempt_data = {
                    "attempt": attempt,
                    "tool_used": tool_used,
                    "execution_success": exec_success,
                    "time": time.time() - attempt_start_time
                }
                
                query_result["attempts"].append(attempt_data)
                
                # Update message history for next attempt
                message_history.append({"role": "user", "content": prompt})
                message_history.append({
                    "role": "assistant", 
                    "content": response.get("message", {}).get("content", ""),
                    "tool_calls": response.get("message", {}).get("tool_calls", [])
                })
                
                if tool_used and exec_success:
                    print(f"\nSuccessful execution on attempt {attempt}!")
                    query_result["success"] = True
                    query_result["best_attempt"] = attempt
                    break
                
                # If we've used the tool but it wasn't successful, add feedback
                if tool_used and not exec_success:
                    feedback = f"Your command didn't work correctly. Try a different approach. The error was: {tool_content[:200]}..."
                    message_history.append({"role": "user", "content": feedback})
                    
            except Exception as e:
                print(f"Error during test: {str(e)}")
                query_result["attempts"].append({
                    "attempt": attempt,
                    "tool_used": False,
                    "execution_success": False,
                    "error": str(e),
                    "time": time.time() - attempt_start_time
                })
        
        # After all attempts, record the results
        best_attempt = None
        for attempt_data in query_result["attempts"]:
            if attempt_data["tool_used"] and attempt_data["execution_success"]:
                best_attempt = attempt_data
                break
        
        if best_attempt:
            successful_tool_uses += 1
            successful_executions += 1
            
        results["queries"].append(query_result)
        
        print(f"\nCompleted query {i+1}: {'SUCCESS' if query_result['success'] else 'FAILURE'}")
        if query_result["success"]:
            print(f"Successful on attempt {query_result['best_attempt']}/{max_attempts}")
    
    # Calculate rates
    results["tool_usage_rate"] = (successful_tool_uses / len(test_queries)) * 100
    results["success_rate"] = (successful_executions / len(test_queries)) * 100
    
    # Calculate complexity-based results
    complexity_groups = {}
    for query_result in results["queries"]:
        complexity = query_result["complexity"]
        if complexity not in complexity_groups:
            complexity_groups[complexity] = []
        complexity_groups[complexity].append(query_result)
    
    for complexity, queries in complexity_groups.items():
        successful = sum(1 for q in queries if q["success"])
        total = len(queries)
        results["complexity_results"][complexity] = {
            "queries": total,
            "successful": successful,
            "success_rate": (successful / total) * 100 if total > 0 else 0
        }
    
    print(f"\n===== COMPLETED TESTING MODEL: {model_name} =====\n")
    print(f"Tool usage rate: {results['tool_usage_rate']:.1f}% ({successful_tool_uses}/{len(test_queries)} queries)")
    print(f"Success rate: {results['success_rate']:.1f}% ({successful_executions}/{len(test_queries)} queries)")
    
    # Save results
    model_safe_name = model_name.replace(':', '_').replace('/', '_')
    filename = os.path.join(results_dir, f"terminal_test_{model_safe_name}.json")
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")
    
    # Generate a detailed report
    report_filename = os.path.join(results_dir, f"terminal_test_report_{model_safe_name}.txt")
    with open(report_filename, 'w') as f:
        f.write(f"Terminal Command Test Results for {model_name}\n")
        f.write(f"Test Time: {results['test_time']}\n")
        f.write(f"Max Attempts Per Task: {max_attempts}\n")
        f.write(f"Force Tool Usage: {force_tool_usage}\n")
        f.write(f"Tool Usage Rate: {results['tool_usage_rate']:.1f}%\n")
        f.write(f"Success Rate: {results['success_rate']:.1f}%\n\n")
        
        f.write("Complexity Analysis:\n")
        for complexity, data in results["complexity_results"].items():
            f.write(f"  {complexity} Commands: {data['successful']}/{data['queries']} ")
            f.write(f"({data['success_rate']:.1f}%)\n")
        
        f.write("\nDetailed Query Results:\n")
        for i, query_result in enumerate(results['queries'], 1):
            f.write(f"\n{i}. [{query_result['complexity']}] '{query_result['query']}'\n")
            f.write(f"   Success: {'Yes' if query_result['success'] else 'No'}\n")
            
            if query_result["success"]:
                f.write(f"   Successful on attempt: {query_result['best_attempt']}/{max_attempts}\n")
            
            f.write("   Attempts:\n")
            for attempt in query_result["attempts"]:
                f.write(f"     #{attempt['attempt']}: Tool Used: {'Yes' if attempt['tool_used'] else 'No'}, ")
                f.write(f"Success: {'Yes' if attempt.get('execution_success', False) else 'No'}, ")
                f.write(f"Time: {attempt['time']:.2f}s\n")
                if 'error' in attempt:
                    f.write(f"       Error: {attempt['error']}\n")
    
    print(f"Detailed report saved to {report_filename}")
    
    return results

def main():
    """Main function to test a single model with terminal commands."""
    print("Improved Terminal Command Testing Script")
    print("Testing model's ability to handle terminal commands with multiple attempts")
    
    # Model to test
    model_name = "spratling/mistral-small-3.1-24B-it-2503:Q8_0"  # Replace with your chosen model
    # model_name = "qwq:latest"
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"terminal_test_improved_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Saving results to directory: {results_dir}")
    
    # Run the test with multiple attempts and forced tool usage
    results = run_test_scenario(model_name, results_dir, max_attempts=5, force_tool_usage=True)
    
    # Display complexity-based analysis
    print("\n===== COMPLEXITY ANALYSIS =====\n")
    
    for complexity, data in results["complexity_results"].items():
        print(f"{complexity} Commands:")
        print(f"  Success Rate: {data['success_rate']:.1f}% ({data['successful']}/{data['queries']})")
        
        # Calculate average attempts for successful queries
        successful_queries = [q for q in results["queries"] 
                             if q["complexity"] == complexity and q["success"]]
        if successful_queries:
            avg_attempts = sum(q["best_attempt"] for q in successful_queries) / len(successful_queries)
            print(f"  Average attempts until success: {avg_attempts:.1f}")
        
    print("\n===== TEST COMPLETE =====\n")

if __name__ == "__main__":
    main()