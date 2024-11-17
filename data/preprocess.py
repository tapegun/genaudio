import numpy as np
import librosa

def preprocess_audio(file_path, sample_rate=16000, chunk_size=16000):
    audio, _ = librosa.load(file_path, sr=sample_rate)
    # Discretize audio to a fixed number of levels
    audio = np.clip(((audio + 1) * 127.5).astype(np.int32), 0, 255)
    # Normalize to float range [0, 1]
    audio = audio / 255.0
    # Split into chunks
    chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]
    return chunks

