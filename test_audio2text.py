import os
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src import laion_clap
import numpy as np


models = [
    # "base"
    # "reweighting_5_1e3_acaps_clotho",
    # "reweighting_5_1e4_acaps_clotho",
    # "reweighting_7_1e2_acaps_clotho",
    # "reweighting_7_1e2_acaps_clotho_30",
    # "reweighting_7_1e3_acaps_clotho",
    # "reweighting_7_1e3_acaps_clotho_30",
    # "reweighting_7_1e4_acaps_clotho",
    # "reweighting_9_1e3_acaps_clotho",
    # "reweighting_9_1e4_acaps_clotho",
    "base",
    "reweighting_0_1e5_acaps_clotho",
    "reweighting_1_1e5_acaps_clotho",
    "reweighting_2_1e5_acaps_clotho",
    "reweighting_3_1e5_acaps_clotho",
    "reweighting_4_1e5_acaps_clotho",
    "reweighting_5_1e5_acaps_clotho",
    "reweighting_6_1e5_acaps_clotho",
    "reweighting_7_1e5_acaps_clotho",
    "reweighting_8_1e5_acaps_clotho",
    "reweighting_9_1e5_acaps_clotho",
    "reweighting_10_1e5_acaps_clotho",
    "reweighting_11_1e5_acaps_clotho",
]

# Ensure CUDA is available
print(f"CUDA Available: {torch.cuda.is_available()}")

# Define device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load dataset
dataset_path = 'data/audiocaps/test.csv'
data = pd.read_csv(dataset_path)

# Add ground truth column (caption corresponding to each audio filename)
data['ground_truth_caption'] = data['caption'].tolist()
ground_truth_captions = data['ground_truth_caption']


for model_name in models:
    print("Evaluation for", model_name)
    # Load the model
    model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
    if model_name == "base":
        model.load_ckpt('models/630k-audioset-best.pt', strict=False)
    else:
        model.load_ckpt(f'logs/{model_name}/checkpoints/epoch_latest.pt', strict=False)

    # Define audio folder
    audio_folder = "data/audiocaps/audio_files/test"
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
    # print('audio_files')
    # print(audio_files)

    full_audio_files_paths = [os.path.join(audio_folder, file_name) for file_name in audio_files]
    # print('full_audio_files_paths')
    # print(full_audio_files_paths)

    # # Load an example waveform to validate audio files
    # import torchaudio
    # waveform, sample_rate = torchaudio.load(full_audio_files_paths[0])
    # print('waveform.shape')
    # print(waveform.shape)

    # Process captions
    captions = data['caption'].tolist()

    # Check if torch.amp.autocast is available and compatible
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        use_amp = True
    else:
        use_amp = False

    # Generate embeddings with AMP (PyTorch 1.11+)
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):  # Use device_type instead of 'cuda'
                print("Generating embeddings with AMP...")
                audio_embed = model.get_audio_embedding_from_filelist(full_audio_files_paths)
                text_embed = model.get_text_embedding(captions)
        else:
            print("Generating embeddings without AMP...")
            audio_embed = model.get_audio_embedding_from_filelist(full_audio_files_paths)
            text_embed = model.get_text_embedding(captions)

    # Ensure text_embed and audio_embed are PyTorch tensors (convert from NumPy arrays if needed)
    if isinstance(text_embed, np.ndarray):
        text_embed = torch.tensor(text_embed).to(device)  # Convert to PyTorch tensor and move to device

    if isinstance(audio_embed, np.ndarray):
        audio_embed = torch.tensor(audio_embed).to(device)  # Convert to PyTorch tensor and move to device

    # Handle NaN values by replacing them with zero (or another suitable value)
    text_embed = torch.nan_to_num(text_embed, nan=0.0)
    audio_embed = torch.nan_to_num(audio_embed, nan=0.0)

    # Compute cosine similarity and top matches for Audio-to-Text
    audio_to_text_matches = []
    for i, audio_path in enumerate(full_audio_files_paths):
        audio_embedding = audio_embed[i].reshape(1, -1)
        similarity_scores = cosine_similarity(audio_embedding.cpu(), text_embed.cpu())[0]
        top_indices = similarity_scores.argsort()[::-1][:3]
        top_matches = [captions[idx] for idx in top_indices]

        # Check if ground truth is in the top matches
        ground_truth = ground_truth_captions[i]
        is_ground_truth_present = any(ground_truth == caption for caption, _ in top_matches)

        audio_to_text_matches.append({
            "audio_file": os.path.basename(audio_path),
            "top_matches": top_matches,
            "ground_truth": ground_truth,
            "is_ground_truth_present": is_ground_truth_present
        })

    # # Display results for Audio-to-Text Matches
    # for match in audio_to_text_matches:
    #     print(f"\nAudio File: {match['audio_file']}")
    #     print(f"Ground Truth: {match['ground_truth']}")
    #     print(f"Ground Truth in Top Matches: {match['is_ground_truth_present']}")
    #     for i, (caption, score) in enumerate(match['top_matches'], start=1):
    #         print(f"  {i}. {caption} (Score: {score:.2f})")

    # Calculate accuracy
    top_1_correct = sum(1 for match in audio_to_text_matches if match['ground_truth'] == match['top_matches'][0])
    top_5_correct = sum(1 for match in audio_to_text_matches if match['ground_truth'] in match['top_matches'][:5])
    top_10_correct = sum(1 for match in audio_to_text_matches if match['ground_truth'] in match['top_matches'])

    total = len(audio_to_text_matches)
    top_1_accuracy = top_1_correct / total
    top_5_accuracy = top_5_correct / total
    top_10_accuracy = top_10_correct / total

    print(f"\nTop-1 Accuracy: {top_1_accuracy:.2%}")
    print(f"Top-5 Accuracy: {top_5_accuracy:.2%}")
    print(f"Top-10 Accuracy: {top_10_accuracy:.2%}")

    print()
    print()
