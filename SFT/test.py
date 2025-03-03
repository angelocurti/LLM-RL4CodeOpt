import os
import sys
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import peft
from peft import get_peft_model, LoraConfig
import csv
import json
import argparse
from huggingface_hub import login

###HUGGING FACE LOG IN
login(token='...')

parser = argparse.ArgumentParser()

## PARSING ARGUMENTS  
parser.add_argument("--dataset_path", default="Dataset/Test_Dataset.json", type=str, help="path to load dataset")  
parser.add_argument("--output_path", default=None, type=str, help="output directory")
parser.add_argument("--model_path", default=None, type=str, help="path to load models")
parser.add_argument("--max_source_length", default=256, type=int, help="maximum source length")
parser.add_argument("--max_target_length", default=256, type=int, help="maximum target lengt")

print("STARTED TESTING")
###MODEL LOADING
print("LOADING FINE-TUNED MODEL")
model = AutoModelForCausalLM.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, truncation=True, padding='max_length', max_length=512)
print("FINE-TUNED MODEL OK")

###TEST SET LOADING
with open('Dataset/Test_Dataset.json', 'r') as f:
    dataset = json.load(f)

# helper to test the model on a code snippet
def test_model_on_code(code):

    inputs = tokenizer(code, truncation=True, padding='max_length', max_length=args.max_source_length, return_tensors="pt")
    
    outputs = model.generate(inputs['input_ids'].to(device), attention_mask=inputs['attention_mask'].to(device), max_new_tokens=args.max_target_length)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

###TESTING
results = []
i = 0
prompt = "Below there is a slow code version.\n Optimize it both in execution time and in memory consumption using Numoy and Math libraries.\n\n### Input:\n{input}\n\n### Optimized version:"
print("STARTING TESTS")
# cycle over the test dataset
for example in dataset:
    helper = example['code']
    code = prompt.format(input= helper)
    print(i)
    # response from the model
    generated_output = test_model_on_code(code)
    # dictionary to store the results
    result = {
        'title': example['title'],
        'code': code,
        'generated_output': generated_output
    }
    
    results.append(result)

print("TEST OK")
# save the results in a json file
with open('generated_outputs.json', 'w') as f:
    json.dump(results, f, indent=4)

print("END TESTING")
