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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Load dataset
dataset_path = 'data/audiocaps/test.csv'
audio_folder = "data/audiocaps/audio_files/test"
data = pd.read_csv(dataset_path)

# Add ground truth column (audio filename corresponding to each caption)
data['ground_truth_audio'] = data.apply(
    lambda row: f"{row['youtube_id']}_{row['start_time']}.wav", axis=1
)
ground_truth_audio = data['ground_truth_audio'].tolist()
captions = data['caption'].tolist()

# Define audio folder
audio_files_paths = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith('.wav')]


for model_name in models:
    print("Evaluation for", model_name)
    # Load the model
    model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
    if model_name == "base":
        model.load_ckpt('models/630k-audioset-best.pt', strict=False)
    else:
        model.load_ckpt(f'logs/{model_name}/checkpoints/epoch_latest.pt', strict=False)

    # Generate embeddings with autocast for compatibility
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            print("Generating embeddings...")
            text_embed = model.get_text_embedding(captions)
            audio_embed = model.get_audio_embedding_from_filelist(audio_files_paths)

    # print(text_embed.shape, audio_embed.shape)

    # Ensure text_embed and audio_embed are PyTorch tensors
    if isinstance(text_embed, np.ndarray):
        text_embed = torch.tensor(text_embed).to(device)
    if isinstance(audio_embed, np.ndarray):
        audio_embed = torch.tensor(audio_embed).to(device)

    # Handle NaN values by replacing them with zero
    text_embed = torch.nan_to_num(text_embed, nan=0.0)
    audio_embed = torch.nan_to_num(audio_embed, nan=0.0)

    # Compute cosine similarity and top matches for Text-to-Audio
    text_to_audio_matches = []
    for i, caption in enumerate(captions):
        text_embedding = text_embed[i].reshape(1, -1)
        similarity_scores = cosine_similarity(text_embedding.cpu(), audio_embed.cpu())[0]
        top_indices = similarity_scores.argsort()[::-1][:10]
        top_matches = [os.path.basename(audio_files_paths[idx]) for idx in top_indices]

        #print(sorted(similarity_scores)[::-1][:10])

        # Check if ground truth is in the top matches
        ground_truth = ground_truth_audio[i]
        is_ground_truth_present = any(ground_truth == os.path.basename(audio) for audio in top_matches)

        text_to_audio_matches.append({
            "caption": caption,
            "top_matches": top_matches,
            "ground_truth": ground_truth,
            "is_ground_truth_present": is_ground_truth_present
        })

    # # Display results for Text-to-Audio Matches
    # for i, match in enumerate(text_to_audio_matches):
    #     if i == 5: break
    #     print(f"\nCaption: {match['caption']}")
    #     print(f"Ground Truth: {match['ground_truth']}")
    #     print(f"Ground Truth in Top Matches: {match['is_ground_truth_present']}")
    #     for i, (audio_file, score) in enumerate(match['top_matches'], start=1):
    #         print(f"  {i}. {audio_file} (Score: {score:.2f})")


    # Calculate accuracy
    top_1_correct = sum(1 for match in text_to_audio_matches if match['ground_truth'] == match['top_matches'][0])
    top_5_correct = sum(1 for match in text_to_audio_matches if match['ground_truth'] in match['top_matches'][:5])
    top_10_correct = sum(1 for match in text_to_audio_matches if match['ground_truth'] in match['top_matches'])

    total = len(text_to_audio_matches)
    top_1_accuracy = top_1_correct / total
    top_5_accuracy = top_5_correct / total
    top_10_accuracy = top_10_correct / total

    print(f"\nTop-1 Accuracy: {top_1_accuracy:.2%}")
    print(f"Top-5 Accuracy: {top_5_accuracy:.2%}")
    print(f"Top-10 Accuracy: {top_10_accuracy:.2%}")

    print()
    print()
