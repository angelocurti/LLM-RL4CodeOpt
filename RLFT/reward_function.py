import numpy as np
import re
import torch
import sys
import time
import subprocess
    
def remove_special_tokens(code_string):
    lines = code_string.split("NEW_LINE")
    lines = [item.strip() for item in lines]
    
    curr_indent = 0
    new_lines = []
    for line in lines:
        indent_count = line.count('INDENT')
        dedent_count = line.count('DEDENT')
        curr_indent += indent_count - dedent_count
        wo_indent = re.sub('INDENT\s?', '', line)
        wo_dedent = re.sub('DEDENT\s?', '', wo_indent)
        new_lines.append('\t'*curr_indent + wo_dedent)
    return ("\n").join(new_lines)

def execute_code(code, lang):
    import time
import subprocess
import psutil

def measure_script_performance(script_code, input_path, expected_output_path, timeout_seconds=15, verbose=False):
    
    # test case loading
    #input
    with open(input_path, "r") as input_f:
        input_cases = input_f.readlines()
    
    #expected output
    with open(expected_output_path, "r") as expected_f:
        expected_outputs = expected_f.readlines()
    
    # Check if the number of test cases matches the number of expected outputs
    if len(input_cases) != len(expected_outputs):
        print("Errore: il numero di casi di test non corrisponde al numero di output attesi.")
        return None
    
    script_file = "temp_script.py"  # Percorso al file temporaneo
    with open(script_file, "w", encoding='utf-8') as f:
        f.write(script_code)
    
    max_duration = 0  # Tempo massimo di esecuzione
    max_memory_usage = 0  # Consumo massimo di memoria
    all_correct = True

    for i, (input_case, expected_output) in enumerate(zip(input_cases, expected_outputs)):
        # Remove leading/trailing whitespaces
        input_case = input_case.strip()
        expected_output = expected_output.strip()

        # write input to a temporary file
        with open("temp_input.txt", "w") as temp_input_file:
            temp_input_file.write(input_case)

        # Esegui il caso di test
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
                        # Comunica con il processo per catturare stdout e stderr
                        stdout, stderr = proc.communicate(timeout=timeout_seconds)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        stdout, stderr = b"", b"Timeout Expired"

        end = time.perf_counter()

        # Calcolo delle metriche
        duration = end - start - 0.1
        script_output = stdout.decode("utf-8").strip()

        # max duration and memory
        max_duration = max(max_duration, duration)
        max_memory_usage = max(max_memory_usage, max_memory)
        #checking functional correctness
        is_correct = script_output == expected_output
        if verbose:
            if stderr:
                print("Captured stderr (errors):")
            
            if not is_correct:
                all_correct = False
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
    
    return max_duration, max_memory_usage, all_correct



def get_reward(test_cases = None, code_ids=None, base_ids=None, tokenizer=None):
    code_ids = np.array(code_ids.cpu())
    eos_positions = []
    max_len = code_ids.shape[1]
    for id in code_ids:
        if tokenizer.eos_token_id in id:
            eos_positions.append((id==tokenizer.eos_token_id).argmax())
        else:
            eos_positions.append(max_len)

    codes = [tokenizer.decode(id[:eos_pos], skip_special_tokens=True, clean_up_tokenization_spaces=False) \
             for id,eos_pos in zip(code_ids, eos_positions)]
    codes_base = [tokenizer.decode(id[:eos_pos], skip_special_tokens=True, clean_up_tokenization_spaces=False) \
             for id,eos_pos in zip(base_ids, eos_positions)] 
    if (test_cases != None):    
        metrics = [
            measure_script_performance(code, test_case.input_path, test_case.expected_output_path) 
            for code, test_case in zip(codes, test_cases)
        ]
        base_metrics = [
            measure_script_performance(code, test_case.input_path, test_case.expected_output_path) 
            for code, test_case in zip(codes_base, test_cases)
        ]
    
    rewards = np.zeros_like(code_ids, dtype=float)
    compile_batch = 0
    execution_time_batch = 0
    memory_usage_batch = 0
    for i in range(len(rewards)):
        if (test_cases != None):
            execution_time, memory_usage, did_compile = metrics[i]
            execution_time_base, memory_time_base, did_compile_base = base_metrics[i]
        else:
            execution_time, memory_usage, did_compile = 0.2, 0.1, 1
            execution_time_base, memory_time_base, did_compile_base = 0.3, 0.2, 1
        reward = 1 if did_compile else -1

        compile_batch += reward
        execution_time_batch += execution_time
        memory_usage_batch += memory_usage
        rewards[i, min(eos_positions[i],max_len-1)] = reward + (execution_time_base- execution_time) + (memory_time_base-memory_usage)
        #aggiungere penalità tipo ast per mismatch con la soluzione esatta
     
    mean_rate = compile_batch/len(codes)
    mean_execution_time = execution_time_batch/len(codes)
    mean_memory_usage = memory_usage_batch/len(codes)
    
    return torch.Tensor(rewards),mean_rate,mean_execution_time,mean_memory_usage