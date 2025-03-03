import numpy as np
import re
import torch
import sys
import time
import subprocess
import time
import subprocess
import psutil
import os  

def measure_script_performance(code, test_path, timeout_seconds=5, verbose=False):
    
    # test case loading
    inputs = os.path.join(test_path, "input.0.txt")
    expected_outputs = os.path.join(test_path, "output.0.txt")
    if not os.path.exists(inputs) or not os.path.exists(expected_outputs):
        print(f"Error: invalid path\n")
        return 0, 0, False
    
    #reading input and output test cases
    with open(inputs, "r") as input_f:
        input_cases = input_f.readlines()

    with open(expected_outputs, "r") as expected_f:
        expected_output = expected_f.readlines()
    
    # common imports to avoid that programs fails too much 
    common_imports = """
import os
import sys
import math
import re
import time
import random
import datetime
import json
import csv
import numpy as np
import pandas as pd
from collections import defaultdict, Counter, deque
from itertools import permutations, combinations, product
"""
    
    #script with needed imports
    program = common_imports + "\n" + code
    
    #saving it to a temporary file
    script_file = "temp_script.py"  
    with open(script_file, "w", encoding='utf-8') as f:
        f.write(program)
    
    #initializing future results
    max_duration = 0  
    max_memory_usage = 0 
    all_correct = True

    for i, (input_case, expected_output) in enumerate(zip(input_cases, expected_output)):
        
        #remove leading/trailing whitespaces
        input_case = input_case.strip()
        expected_output = expected_output.strip()

        #write input to a temporary file
        with open("temp_input.txt", "w") as temp_input_file:
            temp_input_file.write(input_case)

        ###CODE EXECUTION
        start = time.perf_counter()
        
        with open("temp_input.txt", "r") as temp_input_f:
            with subprocess.Popen(
                ["python", script_file],
                stdin=temp_input_f,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            ) as proc:
                # memory monitoring
                process = psutil.Process(proc.pid)
                max_memory = 0
                timeout_reached = False
                
                while proc.poll() is None:
                    # timeout check
                    elapsed_time = time.perf_counter() - start
                    if elapsed_time > timeout_seconds:
                        proc.kill()  # stop the process
                        timeout_reached = True
                        break
                        
                    try:
                        # memory
                        max_memory = max(max_memory, process.memory_info().rss)
                    except psutil.NoSuchProcess:
                        break

                    time.sleep(0.1)  # briefly sleep to avoid 100% CPU usage
                
                if timeout_reached:
                    stdout, stderr = b"", b"Timeout Expired"
                else:
                    try:
                        #debugging
                        stdout, stderr = proc.communicate(timeout=timeout_seconds)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        stdout, stderr = b"", b"Timeout Expired"

        end = time.perf_counter()

        ###METRICS
        duration = end - start - 0.1
        script_output = stdout.decode("utf-8").strip()

        # max duration and memory
        max_duration = max(max_duration, duration)
        max_memory_usage = max(max_memory_usage, max_memory)
        #checking functional correctness
        all_correct = script_output == expected_output

        if verbose:
            if stderr:
                print("Captured stderr (errors):")
            
            if not is_correct:
                print(f"Test case {i + 1} FAILED!")
                print(f"  Input: {input_case}")
                print(f"  Expected: {expected_output}")
                print(f"  Got: {script_output}")
            else:
                print(f"Test case {i + 1} PASSED.")

            # Final Statistics
            print(f"\nTempo massimo di esecuzione: {max_duration:.4f} secondi")
            print(f"Consumo massimo di memoria: {max_memory_usage / 1024**2:.4f} MB")
            print(f"Accuracy complessiva: {'Correct' if all_correct else 'Incorrect'}")
    
    # Cleanup temp files
    try:
        os.remove(script_file)
        os.remove("temp_input.txt")
    except:
        pass
        
    return max_duration, max_memory_usage, all_correct


def get_reward(test_cases, code_ids, base_codes, codes, tokenizer = None):
    
    """This function calculates the reward derived from execution of a batch of snippet of code. 
    Reward is calculated considering functional correctness, execution time and memory usage improvement."""
    
    #initiliazation
    code_ids = np.array(code_ids.cpu()) #to be compliant with kl reward
    max_len = code_ids.shape[1]
    metrics = []
    metrics_base = []
    rewards = np.zeros_like(code_ids, dtype=float)
    compile_batch = 0
    execution_time_batch = 0
    memory_usage_batch = 0
    
    #padding
    for id in code_ids:
        if tokenizer.eos_token_id in id:
            eos_positions.append((id==tokenizer.eos_token_id).argmax())
        else:
            eos_positions.append(max_len)
    
    #execute base code and improved versions
    for i in range(len(test_cases)):
        if (codes[i]):
            metrics.append(measure_script_performance(codes[i], f"test_cases/public_test_cases/{test_cases[i]}"))
        else:
            metrics.append(0,0, False)
        if(base_codes[i]):
            metrics_base(measure_script_performance(base_codes[i], f"test_cases/public_test_cases/{test_cases[i]}"))
        else:
            metrics_base.append(0,0, False)

    #calculating reward tensor
    for i in range(len(rewards)):
        
        execution_time, memory_usage, did_compile = metrics[i][0], metrics[i][1], metrics[i][2]
        execution_time_base, memory_usage_base = base_perf[i][0], base_perf[i][1]
        execution_time_batch += execution_time
        memory_usage_batch += memory_usage
        #checking if time and memory usage have been improved
        reward = 1 if (execution_time < execution_time_base and memory_usage < memory_usage_base) else (0.5 if (execution_time < execution_time_base) else -0.5)
        reward = reward if did_compile else -1
        compile_batch += reward
        #rewards tensor
        rewards[i, min(eos_positions[i],max_len-1)] = reward
    
    #statistics
    mean_rate = compile_batch/len(codes) if (codes) else 0
    mean_execution_time = execution_time_batch/len(codes) if (codes) else 0
    mean_memory_usage = memory_usage_batch/len(codes) if (codes) else 0
    
    return torch.Tensor(rewards),mean_rate,mean_execution_time,mean_memory_usage