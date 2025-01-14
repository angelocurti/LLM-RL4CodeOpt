from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@dataclass
class ModelArguments:
    """
    Arguments for configuring and loading the model and tokenizer used in training.
    """
    model_name_or_path: str = field(
        default="Salesforce/codet5p-770m",
        metadata={"help": "Path or name of the pre-trained model to fine-tune."}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path or name of the tokenizer to use. Defaults to model_name_or_path."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a directory where the model/tokenizer cache should be stored."}
    )

    def load_model_and_tokenizer(self):
        """
        Load the model and tokenizer based on the provided paths or names.
        """
        model_path = self.model_name_or_path
        tokenizer_path = self.tokenizer_name_or_path or model_path

        print(f"Loading model from: {model_path}")
        print(f"Loading tokenizer from: {tokenizer_path}")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=self.cache_dir)   #if the architecture changes this line must be changed
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, cache_dir=self.cache_dir)   #if the architecture changes this line must be changed

        return model, tokenizer
