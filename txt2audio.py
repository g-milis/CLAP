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

# Compute audio embeddings for all audio files
audio_embeddings = []
audio_files = []

print("Computing audio embeddings...")
for audio_file in os.listdir(AUDIO_FOLDER_PATH):
    if audio_file.endswith(".wav"):  # Ensure only .wav files are processed
        audio_path = os.path.join(AUDIO_FOLDER_PATH, audio_file)
        embedding = get_audio_embeddings(audio_path)
        audio_embeddings.append(embedding)
        audio_files.append(audio_file)

audio_embeddings = np.vstack(audio_embeddings)  # Combine all embeddings into a single array

# Compute similarities for each text caption
text_to_audio_matches = []
print("Computing text-to-audio retrieval...")
text_embeddings = get_text_embeddings(captions)

for i, caption in enumerate(captions):
    # Compute cosine similarity between text embedding and all audio embeddings
    text_embedding = text_embeddings[i].reshape(1, -1)
    similarity_scores = cosine_similarity(text_embedding, audio_embeddings)[0]

    # Get top 3 matches
    top_indices = np.argsort(similarity_scores)[::-1][:3]
    top_matches = [(audio_files[idx], similarity_scores[idx]) for idx in top_indices]

    text_to_audio_matches.append({
        "caption": caption,
        "top_matches": top_matches
    })

# Display results
print("\nTop Matches for Each Text Caption:")
for match in text_to_audio_matches:
    print(f"\nCaption: {match['caption']}")
    for i, (audio_file, score) in enumerate(match['top_matches'], start=1):
        print(f"  {i}. {audio_file} (Score: {score:.2f})")
