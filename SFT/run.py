#Loading needed libraries 
import transformers
from transformers import TrainingArguments as HFTrainingArguments
from transformers import Trainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dataclasses import dataclass, field
from typing import Optional
import sys
import os
import torch
from utils import ModelArguments, TrainingArguments, DynamicTrainer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Dataset')))
import DataLoader

# Dataset and model
dataset_path = "Dataset/Code_Pairs.csv"  
model_path = "Salesforce/codet5p-770m"  

model_args = ModelArguments()
model, tokenizer = model_args.load_model_and_tokenizer()
# Tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained(model_path, truncation=True, padding='max_length', max_length=512)

# Load the dataset and tokenize it
data_loader = DataLoader.DataLoader()
data_loader.load(dataset_path)
data_loader.tokenize(tokenizer, max_length = 256)

#splitting into train and test
data_loader.split_data()

# Training arguments definition
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=6,
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",
)

# Trainer initialization
trainer = DynamicTrainer(
    model=model,
    args=training_args,
    train_dataset=data_loader.tokenized_dataset["train"],
    eval_dataset=data_loader.tokenized_dataset["test"]
)

# Training
trainer.train()

#load fine-tuned model and test
checkpoint_path = 'results/checkpoint-9438'

# loading model and using the same tokenizer as before
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
#saving the model to better usability
model.save('/content/LLM-RL4CodeOpt/ft_model')

import json

# load test dataset
with open('Dataset/Test_Dataset.json', 'r') as f:
    dataset = json.load(f)

# helper to test the model on a code snippet
def test_model_on_code(code):

    inputs = tokenizer(code, return_tensors="pt")
    
    outputs = model.generate(inputs['input_ids'], max_length = 512)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

results = []
i = 0
# cyclo over the test dataset
for example in dataset:
    code = example['code']
    print(i)
    i+=1
    
    # response from the model
    generated_output = test_model_on_code(code)
    
    # dictionary to store the results
    result = {
        'title': example['title'],
        'code': code,
        'generated_output': generated_output
    }
    
    results.append(result)

# save the results in a json file
with open('generated_outputs.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Generazioni salvate nel file 'generated_outputs.json'")
