import os
import re
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src import laion_clap


def batch_process_files(file_list, batch_size, func):
    results = []
    for i in range(0, len(file_list), batch_size):
        batch = file_list[i:i + batch_size]
        batch_result = func(batch)
        results.append(batch_result)
    return np.concatenate(results, axis=0)


def contains_keywords(text, keywords):
    """
    Checks if a string contains any of the specified keywords (case-insensitive).
    """
    pattern = r'\b(' + '|'.join(keywords) + r')\b' # Create a regex pattern for whole words
    return bool(re.search(pattern, text, re.IGNORECASE))


frequent_audio_events = {
    "Birds_and_Traffic": {
        "events": ["Birds Chirping/Singing", "Traffic Noise"],
        "keywords_1": ["bird", "sing", "tweet"],
        "keywords_2": ["traffic", "car", "vehicle"]
    },
    "Speech_and_Music": {
        "events": ["People Talking/Speech", "Background Music"],
        "keywords_1": ["talk", "speak", "laugh"],
        "keywords_2": ["music", "song", "instrument"]
    },
    "Water_and_Birdsong": {
        "events": ["Water Sounds (dripping, flowing)", "Birds Chirping/Singing"],
        "keywords_1": ["water", "drip", "flow"],
        "keywords_2": ["bird", "sing", "tweet"]
    }
}

generic_captions = {
    "Birds_and_Traffic": [
        "Birds are chirping while vehicles pass on a nearby road.",
        "Traffic rumbles in the distance as birds sing in the background."
    ],
    "Speech_and_Music": [
        "People are talking with music playing faintly in the background.",
        "Conversations mingle with music in an open area."
    ],
    "Water_and_Birdsong": [
        "Water is flowing with birds singing intermittently.",
        "Birds chirp near the steady sound of running water."
    ]
}


def find_cooccurring_audio(filename, frequent_audio_events):
    cooccurring_entries = {}

    with open(filename) as f:
        next(f)  # Skip the header line
        for line in f:
            parts = line.strip().split(",", 1)
            file_name = parts[0]
            captions_str = parts[1]
            captions = captions_str.split(",")

            for category, event_data in frequent_audio_events.items():
                # Check if ANY keywords from EACH set are present in ANY caption
                keywords_found = [False, False]
                for i in [1, 2]:
                    keywords = event_data[f"keywords_{i}"]
                    for caption in captions:
                        pattern = r'\b(' + '|'.join(keywords) + r')\b'  # Word boundary
                        if re.search(pattern, caption, re.IGNORECASE):
                            keywords_found[i-  1] = True
                            break
                # If all events have at least one keyword match:
                if all(keywords_found):
                    if category not in cooccurring_entries:
                        cooccurring_entries[category] = []
                    cooccurring_entries[category].append({
                        "file_name": file_name,
                        "captions": captions
                    })

    # Print findings
    for category, files in cooccurring_entries.items():
        print(f"Category: {category}")
        for file in files:
            print(f'"/fs/cbcb-scratch/milis/data/clotho_dataset/evaluation/{file["file_name"]}"')
            for caption in file["captions"]:
                print(caption)
            print()
        print()
        print()


if __name__ == "__main__":
    # find_cooccurring_audio("/fs/cbcb-scratch/milis/data/clotho_dataset/clotho_captions_evaluation.csv", frequent_audio_events)

    models = [
        "base"
    ]

    # Ensure CUDA is available
    print(f"CUDA Available: {torch.cuda.is_available()}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # Load dataset
    dataset_path = "/fs/cbcb-scratch/milis/data/clotho_dataset/clotho_captions_evaluation.csv"
    audio_folder = "/fs/cbcb-scratch/milis/data/clotho_dataset/evaluation"
    data = pd.read_csv(dataset_path)

    # Define audio folder
    audio_files_paths = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith('.wav')]
    ground_truth_audio_files = [os.path.basename(path) for path in audio_files_paths]

    for model_name in models:
        print("Evaluation for", model_name)
        # Load the model
        model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
        if model_name == "base":
            model.load_ckpt('models/630k-audioset-best.pt', strict=False)
        else:
            model.load_ckpt(f'logs/{model_name}/checkpoints/epoch_latest.pt', strict=False)

        # Get all audio embeddings
        # audio_embed = model.get_audio_embedding_from_filelist(audio_files_paths)
        audio_embed = batch_process_files(audio_files_paths, 128, model.get_audio_embedding_from_filelist)

        if isinstance(audio_embed, np.ndarray):
            audio_embed = torch.tensor(audio_embed).to(device)
        audio_embed = torch.nan_to_num(audio_embed, nan=0.0)

        for category, captions in generic_captions.items():
            print("Category:", category)

            # Generate embeddings with autocast for compatibility
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    print("Generating embeddings...")
                    text_embed = model.get_text_embedding(captions)

            if isinstance(text_embed, np.ndarray):
                text_embed = torch.tensor(text_embed).to(device)
            text_embed = torch.nan_to_num(text_embed, nan=0.0)

            # Compute cosine similarity and top matches for Text-to-Audio
            text_to_audio_matches = []
            for i, caption in enumerate(captions):
                text_embedding = text_embed[i].reshape(1, -1)
                similarity_scores = cosine_similarity(text_embedding.cpu(), audio_embed.cpu())[0]
                top_indices = similarity_scores.argsort()[::-1][:10]
                top_matches = [os.path.basename(audio_files_paths[idx]) for idx in top_indices]

                # Find ground truth
                ground_truth_audio = ground_truth_audio_files[i]
                ground_truth_captions = data.loc[data['file_name'] == ground_truth_audio, "caption_1"].tolist()

                is_ground_truth_present = False

                if ground_truth_captions:  # Ensure ground truth captions were found
                    # Check if ground truth captions contain any of the keywords from the audio_event_categories
                    for k, event_data in frequent_audio_events.items():
                        keywords = []
                        for key in event_data.keys():
                            if "keywords_" in key:
                                keywords += event_data[key]

                        if any(contains_keywords(caption, keywords) for caption in ground_truth_captions):
                                is_ground_truth_present = True
                                break

                text_to_audio_matches.append({
                    "caption": caption,
                    "top_matches": [os.path.basename(path) for path in top_matches],
                    "ground_truth": ground_truth_audio,
                    "is_ground_truth_present": is_ground_truth_present
                })

            # Display results for Text-to-Audio Matches
            for i, match in enumerate(text_to_audio_matches):
                #if i == 5: break
                print(f"\nCaption: {match['caption']}")
                print(f"Ground Truth: {match['ground_truth']}")
                print(f"Ground Truth in Top Matches: {match['is_ground_truth_present']}")
                for j, audio_file in enumerate(match['top_matches']):
                    print(f"  {j+1}. {audio_file}")
