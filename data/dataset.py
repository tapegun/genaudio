import json
import os
import torch
from torch.utils.data import Dataset
from data.preprocess import preprocess_audio  # Assuming preprocess_audio is defined in preprocess.py
from tqdm import tqdm

class AudioDataset(Dataset):
    def __init__(self, json_path="dataset.json", chunk_size=16000):
        with open(json_path, "r") as f:
            data = json.load(f)

        self.data = []
        self.filepaths = []
        for directory, files in tqdm(data.items(), desc="Processing Directories", unit="directory"):
            for file in files:
                file_path = os.path.join(directory, file)
                chunks = preprocess_audio(file_path, chunk_size=chunk_size)
                self.data.extend(chunks)
                self.filepaths.extend(file_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # return self.filepaths[idx], torch.tensor(self.data[idx], dtype=torch.long)
        return torch.tensor(self.data[idx], dtype=torch.float)
