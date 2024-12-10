import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import torchaudio
from src import laion_clap

# Paths to your data
AUDIO_FOLDER_PATH = "data/audiocaps/audio_files/test"  # Path to the folder containing audio files
TEXT_CAPTIONS_CSV_PATH = "data/audiocaps/test.csv"  # Path to your CSV file
MODEL_PATH = "models/630k-audioset-best.pt"  # Path to your saved model

# Load the custom model
model = laion_clap.CLAP_Module()
model.load_ckpt(MODEL_PATH, strict=False)
model.eval()  # Set the model to evaluation mode


# Function to compute text embeddings
def get_text_embeddings(captions):
    inputs = model.tokenize_text(captions)  # Assuming model has a text tokenizer
    with torch.no_grad():
        text_embeddings = model.encode_text(inputs)
    return text_embeddings.numpy()

# Function to compute audio embeddings
def get_audio_embeddings(audio_file_path):
    waveform, sample_rate = torchaudio.load(audio_file_path)
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    waveform = waveform.mean(dim=0).unsqueeze(0)  # Convert to mono
    with torch.no_grad():
        audio_embeddings = model.encode_audio(waveform)
    return audio_embeddings.numpy()

# Load CSV and extract captions
captions_data = pd.read_csv(TEXT_CAPTIONS_CSV_PATH)
captions = captions_data["caption"].tolist()

# Compute text embeddings for all captions
print("Computing text embeddings...")
text_embeddings = get_text_embeddings(captions)

# Process audio files and compute similarities
audio_to_text_matches = []
print("Processing audio files...")
for audio_file in os.listdir(AUDIO_FOLDER_PATH):
    if audio_file.endswith(".wav"):  # Ensure only .wav files are processed
        audio_path = os.path.join(AUDIO_FOLDER_PATH, audio_file)
        audio_id = os.path.splitext(audio_file)[0]  # Get ID without extension

        # Compute audio embedding
        audio_embedding = get_audio_embeddings(audio_path).reshape(1, -1)

        # Compute cosine similarity with all text embeddings
        similarity_scores = cosine_similarity(audio_embedding, text_embeddings)[0]

        # Get top 3 matches
        top_indices = np.argsort(similarity_scores)[::-1][:3]
        top_matches = [(captions[i], similarity_scores[i]) for i in top_indices]

        audio_to_text_matches.append({
            "audio_file": audio_file,
            "top_matches": top_matches
        })

# Display results
print("\nTop Matches for Each Audio File:")
for match in audio_to_text_matches:
    print(f"\nAudio File: {match['audio_file']}")
    for i, (caption, score) in enumerate(match['top_matches'], start=1):
        print(f"  {i}. {caption} (Score: {score:.2f})")
