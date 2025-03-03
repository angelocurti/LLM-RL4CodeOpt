import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class ValueModel(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-3.2-1B", quantization_config=False, dtype=torch.bfloat16):
        super().__init__()
        
        # Set the computation dtype
        self.dtype = dtype
        
        # Use default 4-bit quantization config
        if quantization_config:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model_dim = self.model.config.hidden_size
        self.first_dropout = nn.Dropout(0.1)
        # Initialize the summary layer 
        self.summary = nn.Linear(self.model_dim, 1, dtype=self.dtype)
        
    def load_base_model(self, load_model_path):
        """Load a pre-trained base model from the specified path."""
        self.model.load_state_dict(torch.load(load_model_path))
    
    def forward(self, input_ids, attention_mask=None, labels=None, decoder_attention_mask=None):
        """
        Forward pass through both the language model and value head.
        
        Returns:
            tuple: (logits, full_outputs, value)
        """
        # Standard forward pass
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels,
            use_cache=True,
            output_hidden_states=True
        )
        
        # Get the last hidden states
        hidden_states = outputs.hidden_states[-1].to(self.dtype)
        # Compute the value prediction
        value = self.summary(self.first_dropout(hidden_states)).squeeze(-1)
        # Return the language model outputs, full output object, and value prediction
        return value

