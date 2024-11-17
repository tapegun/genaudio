import torch
from torch.utils.data import DataLoader
from data.dataset import AudioDataset
from models.wavenet import WaveNet
from models.transformer import AudioTransformer
from tqdm import tqdm 

def collate_fn(batch):
    # Find the maximum length in the current batch
    max_length = max(len(tensor) for tensor in batch)
    
    # Pad each tensor to the maximum length
    padded_batch = [torch.nn.functional.pad(tensor, (0, max_length - len(tensor)), mode="constant", value=0) for tensor in batch]
    
    # Stack into a single tensor
    return torch.stack(padded_batch)

def train(model, dataloader, epochs=10, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs), desc="Training Epochs", unit="epoch"):
        for batch in dataloader:
            optimizer.zero_grad()
            # filepath, inputs, targets = batch[0], batch[1][:, :-1], batch[1][:, 1:]
            inputs, targets = batch[:, :-1], batch[:, 1:]
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

if __name__ == "__main__":
    dataset = AudioDataset(json_path="dataset.json")  # Updated to use JSON file
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    model = WaveNet()  # or AudioTransformer()
    train(model, dataloader)
