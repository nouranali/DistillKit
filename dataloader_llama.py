import json
from torch.utils.data import Dataset
from tokenizer import Tokenizer


class LlamaArabicDataset(Dataset):
    def __init__(self, jsonl_file, teacher_model_name):
        self.tokenizer = Tokenizer(teacher_model_name)
        self.data = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item["text"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokenized_data = self.tokenizer.tokenization_function(text)
        return tokenized_data


# # Create dataset
# dataset = LlamaArabicDataset("/ai-workspace/Nouran/model_distillation/model_distillation/data/dataset_wz_transcription.json")

# # Create DataLoader
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# # Example batch
# for batch in dataloader:
#     print(batch["input_ids"].shape)  # Expected: (batch_size, max_length)
#     break
