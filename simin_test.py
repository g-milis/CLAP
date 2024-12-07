import pandas as pd
import laion_clap
from transformers import AutoModel, AutoTokenizer
import torch
#from google.colab import drive
from torch.cuda.amp import autocast
import os
import os
import torchaudio
#!pip install pytube
#from pytube import YouTube
#from moviepy.editor import AudioFileClip

# Mount Google Drive
#drive.mount('/content/drive')
#device = "cpu"

print(torch.cuda.is_available())

# Load dataset from CSV
dataset_path = 'data/audiocaps/test.csv'  # Update with your actual path to CSV
data = pd.read_csv(dataset_path)

# Define device
device = torch.device('cuda:0')

# Load the model and tokenizer
#device = torch.device('cpu')
model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
model.load_ckpt('models/630k-audioset-best.pt')  # Load the model checkpoint if needed
#tokenizer = AutoTokenizer.from_pretrained('roberta-base')  # Adjust if using a different model

# NOTE:
# I use val instead of test

# Path to your folder of audio files
audio_folder = "data/audiocaps/audio_files/val"

# Iterate over all files in the folder
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(('.wav', '.mp3', '.flac'))]

full_audio_files_paths = [os.path.join(audio_folder, file_name) for file_name in audio_files]

# # Load and process each file
# for file_name in audio_files:
#     file_path = os.path.join(audio_folder, file_name)
#     waveform, sample_rate = torchaudio.load(file_path)  # waveform is a Tensor, sample_rate is an int
    
    #print(f"Loaded {file_name}:")

captions = data['caption'].tolist()  # Use captions as the text input
# print("captions:")
# print(captions)
# audio_files = ['path_to_audio_1.wav', 'path_to_audio_2.wav', ...]

# Tokenize captions
#encoded_captions = tokenizer(captions, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    with autocast():
        text_embed = model.get_text_embedding(captions)

        # Get audio embeddings for audio files
        audio_embed = model.get_audio_embedding_from_filelist(full_audio_files_paths)

        # Implement classification or retrieval logic here as per your requirement
        ranking = torch.argsort(torch.tensor(audio_embed) @ torch.tensor(text_embed).t(), descending=True)
        # Continue with ranking or other metrics if needed
