import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class SpamDataset(Dataset):
    """Custom dataset for loading spam classification data."""
    def __init__(self, csv_file: str, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        
        self.max_length = max_length or max(len(x) for x in self.encoded_texts)
        self.encoded_texts = [text[:self.max_length] + [pad_token_id]*(self.max_length-len(text)) for text in self.encoded_texts]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded_texts[idx], dtype=torch.long),
            torch.tensor(self.data.iloc[idx]["Label"], dtype=torch.long)
        )

def get_dataloaders(tokenizer, batch_size: int = 8, num_workers: int = 0):
    """
    Create train, validation, and test dataloaders.
    """
    train_dataset = SpamDataset("train.csv", tokenizer)
    val_dataset = SpamDataset("validation.csv", tokenizer, max_length=train_dataset.max_length)
    test_dataset = SpamDataset("test.csv", tokenizer, max_length=train_dataset.max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_dataset