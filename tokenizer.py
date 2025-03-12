from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, teacher_model_name):
        self.teacher_model_name = teacher_model_name
        self.load_tokenizer()

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.teacher_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Use EOS token as padding

    def tokenization_function(self, examples):
        encoding = self.tokenizer(
            examples,
            truncation=True,
            padding="max_length",  # Ensure consistent sequence length
            max_length=512,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
