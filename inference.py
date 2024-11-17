import torch
from models.wavenet import WaveNet

def generate_audio(model, initial_chunk, length=16000):
    model.eval()
    audio = initial_chunk

    for _ in range(length - len(initial_chunk)):
        input_chunk = torch.tensor(audio[-1], dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            output = model(input_chunk)
        next_sample = output.argmax(dim=-1).item()
        audio.append(next_sample)

    return audio

# Example usage
initial_chunk = [128] * 160  # example starting chunk
model = WaveNet()  # or AudioTransformer()
generated_audio = generate_audio(model, initial_chunk)

