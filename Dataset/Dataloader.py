from datasets import load_dataset
from dataclasses import dataclass, field
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch
"""
This script processes a dataset to perform supervised fine-tuning of a LLM
specifically for the task of code optimization. The script includes functionality to:
1. Load a dataset containing, at least, unoptimized and optimized versions of code.
2. Tokenize the data using a provided prompt template and tokenizer.
3. Split the data into training and validation sets.
4. Retrieve tokenized inputs/outputs for both unoptimized and optimized code.

The dataset is expected to have at least two columns:
  - 'code_unoptimized': Contains the non-optimized version of the code.
  - 'code_optimized': Contains the corresponding optimized version of the code.

The tokenized output will be used to fine-tune models on optimization tasks.
"""

class Dataloader(Dataset):
    prompt: str = ("Below there is a slow code version.\n"
                   "Optimize it both in execution time and in memory consumption.\n\n"
                   "### Input:\n{input}\n\n### Optimized version:")
    dataset: dict = field(default=None)
    tokenized_dataset: dict = field(default=None)
    tokenizer: AutoTokenizer = field(default=None)
    max_length: int = field(default=512)
    
    def __init__(self, dataset, max_length=256):
        """
        Inizializza il DataLoader con il dataset, il tokenizer e la lunghezza massima.
        
        Args:
            dataset (datasets.Dataset or list): Dataset contenente codice non ottimizzato e ottimizzato.
            tokenizer (AutoTokenizer): Tokenizer di Hugging Face per la tokenizzazione del codice.
            max_length (int): Lunghezza massima dei token per l'input.
        """
        self.prompt = ("Below there is a slow code version.\n"
                       "Optimize it both in execution time and in memory consumption.\n\n"
                       "### Input:\n{input}\n\n### Optimized version:")
        self.dataset = dataset
        self.max_length = max_length

    def load(self, data_path: str, dataset_type="csv", shuffle=True):
        """Load a dataset from a file and optionally shuffle it.
           It expects a dataset with at least two columns containing unoptimized and optimized versions of the same code.
        """
        self.dataset = load_dataset(dataset_type, data_files=data_path)
        if shuffle:
            self.dataset = self.dataset.shuffle(seed=42)

    def tokenize(self, tokenizer):
        """Tokenize the dataset using a template prompt and the provided tokenizer."""
        
        def preprocess_function(example):
            # Tokenizza sia l'input che il target
            model_inputs = tokenizer(
                self.prompt.format(input=example['code_unoptimized']),
                return_tensors='pt',# Codice non ottimizzato (input)
                truncation=True,               # Truncamento
                padding='max_length',          # Padding a lunghezza fissa
                max_length=256                 # Lunghezza massima
            )
            
            # Tokenizza l'output (code_optimized)
            #with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                example['code_optimized'],
                return_tensors='pt',# Codice ottimizzato (output)
                truncation=True,             # Truncamento
                padding='max_length',        # Padding a lunghezza fissa
                max_length=256              # Lunghezza massima
            )

            # Aggiungi i target tokenizzati ai dati tokenizzati
            model_inputs['input_ids'] = model_inputs['input_ids'].squeeze(0)
            model_inputs["labels"] = labels["input_ids"].squeeze(0)
            return model_inputs

        # Apply the preprocessing function to the dataset
        self.tokenized_dataset = self.dataset.map(preprocess_function)
        self.tokenized_dataset['train'].remove_columns([col for col in self.tokenized_dataset['train'].column_names if col not in ["input_ids", "attention_mask", "labels"]])
    
    def collate_fn(self):
        """Converte liste in tensori per PyTorch DataLoader."""
        for i in range(len(self.tokenized_dataset['train'])):
            self.tokenized_dataset['train'][i]['input_ids'] = torch.tensor(self.tokenized_dataset['train'][i]['input_ids'])
            self.tokenized_dataset['train'][i]['attention_mask'] = torch.tensor(self.tokenized_dataset['train'][i]['attention_mask'])
            self.tokenized_dataset['train'][i]['labels'] = torch.tensor(self.tokenized_dataset['train'][i]['labels'])

    def split_data(self, test_size=0.1):
        """Split the tokenized dataset into training and validation sets."""
        # Perform the split after tokenizing
        self.tokenized_dataset = self.tokenized_dataset["train"].train_test_split(test_size=test_size)

    def __getitem__(self, idx):
        """
        Recupera un esempio dal dataset e lo tokenizza, restituendo un dizionario con i tensori PyTorch.
        """
        non_optimized_code = self.dataset[idx]['code_unoptimized']
        optimized_code = self.dataset[idx]['code_optimized']

        # Tokenizzazione con Hugging Face Tokenizer
        non_optimized_inputs = self.tokenizer(
            non_optimized_code,
            return_tensors='pt', padding='max_length',
            max_length=self.max_length, truncation=True
        )
        optimized_inputs = self.tokenizer(
            optimized_code,
            return_tensors='pt', padding='max_length',
            max_length=self.max_length, truncation=True
        )
        
        print(f"input_ids type: {type(input_ids)}, shape: {input_ids.shape}")
        print(f"attention_mask type: {type(attention_mask)}, shape: {attention_mask.shape}")
        print(f"label_ids type: {type(label_ids)}, shape: {label_ids.shape}")

        # Ritorna i tensori PyTorch senza la dimensione batch (squeeze(0))
        return {
            'input_ids': non_optimized_inputs['input_ids'].squeeze(0),
            'attention_mask': non_optimized_inputs['attention_mask'].squeeze(0),
            'label_ids': optimized_inputs['input_ids'].squeeze(0)
        }

    def __len__(self):
        return len(self.dataset)
