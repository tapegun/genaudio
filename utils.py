def discretize_audio(audio, levels=256):
    return ((audio + 1) * (levels / 2)).astype(int)

