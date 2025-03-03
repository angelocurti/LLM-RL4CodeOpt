import os
import sys
import torch
import transformers
from transformers import BitsAndBytesConfig as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
import peft
from peft import get_peft_model, LoraConfig, PeftModel
from typing import Optional
import datasets
from datasets import Dataset, load_dataset
from datasets import Dataset, load_dataset
import csv
import json
import argparse
from huggingface_hub import login

###HUGGING FACE LOG IN
login(token='...')

###ARGUMENT PARSING
parser = argparse.ArgumentParser()

## Required parameters  
parser.add_argument("--dataset_path", default="Dataset/Code_Pairs.csv", type=str, help="path to load dataset")  
parser.add_argument("--output_path", default="SFToutput/LLaMa1b/", type=str, help="output directory")
parser.add_argument("--model_path", default="meta-llama/Llama-3.2-1B-Instruct", type=str, help="path to load models")
parser.add_argument("--max_source_length", default=256, type=int, help="maximum source length")
parser.add_argument("--max_target_length", default=256, type=int, help="maximum target length")
parser.add_argument("--train_batch_size", default=16, type=int, help="train_batch_size")
parser.add_argument("--test_batch_size", default=4, type=int,help="test_batch_size")
parser.add_argument("--train_epochs", default=3, type=int,help="test_batch_size")
parser.add_argument("--run", default=1, type=int, help="run ID")
parser.add_argument("--gradient_accumulation_steps", default = 2, type = int)
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")

args = parser.parse_args()
##DATA LOADING
print("STARTED DATA LOADING")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Dataset')))
csv.field_size_limit(2147483647)
#loading
dataset = load_dataset("csv", data_files=args.dataset_path)
#loading tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path, truncation=True, padding="max_length", max_length=args.max_source_length, padding_side = 'left')
tokenizer.pad_token = tokenizer.eos_token

chat_format_data = []

#chat template for LLaMa3.2 dataset
def apply_chat_template(example):
    messages = [
            {"role": "system", "content": "You are a coding assistant that optimizes code to make it more efficient, readable, and maintainable while keeping its functional correctness."},
            {"role": "user", "content": f"Please optimize this code:\n\n```\n{example['code_unoptimized']}\n```"},
            {"role": "assistant", "content": f"Here is the optimized version of your code:\n\n```\n{example['code_optimized']}\n```"}]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt}

#Apply the chat template function to the dataset
new_dataset = dataset.map(apply_chat_template)

#helper for tokenization
def preprocess_for_training(examples):
    return tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=args.max_source_length)

#tokenization
tokenized_dataset = new_dataset.map(preprocess_for_training)#, batched = True)
tokenized_dataset = tokenized_dataset.remove_columns(['code_unoptimized', 'code_optimized', 'problem_id', 'prompt'])
tokenized_dataset.save_to_disk("code_optimization_tokenized_dataset")
tokenized_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1)

#split
train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]
print("DATA LOADING OK")

###MODEL LOADING
print("STARTED MODEL LOADING")

#quantization config
bnb_config = bnb(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.bfloat16,
)

#LLaMa model
model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )

#lora config
lora_config = LoraConfig(
        r=8,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
print("MODEL LOADING OK")

###TRAINING ARGUMENTS
print("SETTING TRAINING ARGUMENTS")

effective_batch_size = args.train_batch_size * args.gradient_accumulation_steps
steps_per_epoch = int(6988*0.9) // effective_batch_size
total_steps = steps_per_epoch * args.train_epochs
warmup_steps = total_steps // 10

training_args = SFTConfig(
    output_dir=args.output_path,    #directory to save the model
    learning_rate=args.lr,
    eval_strategy="steps",  #to evaluate during training
    eval_steps=40,
    per_device_train_batch_size=args.train_batch_size,  #train batch size per GPU
    per_device_eval_batch_size=args.test_batch_size,    #eval batch size per GPU
    seed=0,
    gradient_accumulation_steps=args.gradient_accumulation_steps,#gradient accumulation step
    num_train_epochs=args.train_epochs,
    warmup_steps=warmup_steps,  #warmup per lr scheduler
    lr_scheduler_type="cosine", # scheduler (cosine decay)
    fp16=True,  #mixed precision training
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=400,
    save_total_limit=3,
)
print("TRAINING ARGUMENTS OK")

###SUPERVISED TRAINER
print("INITIALIZING TRAINER")

trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset = tokenized_dataset["test"],
        args=training_args,
        peft_config=lora_config,
    )

print("TRAINER OK")

###RUN FINE-TUNING AND COLLECT RESULT
results = trainer.train()
