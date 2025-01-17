from datasets import load_dataset
from dataclasses import dataclass, field
from transformers import AutoTokenizer
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
@dataclass
class DataLoader:
    prompt: str = ("Below there is a slow code version.\n"
                   "Optimize it both in execution time and in memory consumption.\n\n"
                   "### Input:\n{input}\n\n### Optimized version:")
    dataset: dict = field(default=None)
    tokenized_dataset: dict = field(default=None)
    tokenizer: AutoTokenizer = field(default=None)
    max_length: int = field(default=512)

    def load(self, data_path: str, dataset_type="csv", shuffle=True):
        """Load a dataset from a file and optionally shuffle it.
           It expects a dataset with at least two columns containing unoptimized and optimized versions of the same code.
        """
        self.dataset = load_dataset(dataset_type, data_files=data_path)
        if shuffle:
            self.dataset = self.dataset.shuffle(seed=42)

    def tokenize(self, tokenizer,):
        """Tokenize the dataset using a template prompt and the provided tokenizer."""
        def preprocess_function(example):
            return tokenizer(
                self.prompt.format(input=example['code_unoptimized']),  # Codice non ottimizzato nel prompt
                text_target=example['code_optimized'],                 # Codice ottimizzato come target
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
            )

        # Apply the preprocessing function to the dataset
        self.tokenized_dataset = self.dataset.map(preprocess_function)

    def split_data(self, test_size=0.1):
        """Split the tokenized dataset into training and validation sets."""
        # Perform the split after tokenizing
        self.tokenized_dataset = self.tokenized_dataset["train"].train_test_split(test_size=test_size)

    def __getitem__(self, i):
        # Retrieve the non-optimized and optimized code for the given index `i`
        non_optimized_code = self.dataset['train'][i]['code_unoptimized']
        optimized_code = self.dataset['train'][i]['code_optimized']

        # Use the tokenizer to get input_ids for both codes
        non_optimized_inputs = self.tokenizer(self.prompt.format(input=non_optimized_code), return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True)
        optimized_inputs = self.tokenizer(optimized_code, return_tensors='pt', padding='max_length', max_length=self.max_length)

        # Return a dictionary with `input_ids` for both codes
        return {
            'non_optimized': non_optimized_code,
            'optimized': optimized_code,
            'non_optimized_input_ids': non_optimized_inputs['input_ids'].squeeze(0),  # Remove the batch dimension
            'optimized_input_ids': optimized_inputs['input_ids'].squeeze(0)
        }
