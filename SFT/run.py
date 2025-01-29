#Loading needed libraries 
import transformers
from transformers import TrainingArguments as HFTrainingArguments
from transformers import Trainer
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from typing import Optional
import sys
import os
import torch
from utils import ModelArguments, TrainingArguments, DynamicTrainer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from Dataset import DataLoader

# Dataset and model
dataset_path = "../Datset/Code_pairs.csv"  
model_path = "Salesforce/codet5p-770m"  

model_args = ModelArguments()
model, tokenizer = model_args.load_model_and_tokenizer()
# Tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained(model_path, truncation=True, padding='max_length', max_length=512)

# Load the dataset and tokenize it
data_loader = DataLoader()
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