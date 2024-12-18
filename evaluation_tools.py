"""
From https://github.com/XinhaoMei/audio-text_retrieval/blob/main/tools/utils.py
"""

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


# evaluation tools
def a2t(audio_embs, cap_embs, return_ranks=False):
    # audio to caption retrieval
    num_audios = int(audio_embs.shape[0] / 5)
    index_list = []

    ranks = np.zeros(num_audios)
    top1 = np.zeros(num_audios)
    mAP10 = np.zeros(num_audios)
    for index in range(num_audios):
        # get query audio
        audio = audio_embs[5 * index].reshape(1, audio_embs.shape[1])

        # compute scores
        d = cosine_similarity(audio, cap_embs).squeeze(0).numpy()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        inds_map = []

        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
            if tmp < 10:
                inds_map.append(tmp + 1)
        inds_map = np.sort(np.array(inds_map))
        if len(inds_map) != 0:
            mAP10[index] = np.sum((np.arange(1, len(inds_map) + 1) / inds_map)) / 5
        else:
            mAP10[index] = 0.
        ranks[index] = rank
        top1[index] = inds[0]
    # compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    mAP10 = 100.0 * np.sum(mAP10) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return r1, r5, r10, r50, medr, meanr, ranks, top1, mAP10
    else:
        return r1, r5, r10, r50, medr, meanr, mAP10


def t2a(audio_embs, cap_embs, return_ranks=False):
    # caption to audio retrieval
    num_audios = int(audio_embs.shape[0] / 5)

    audios = np.array([audio_embs[i] for i in range(0, audio_embs.shape[0], 5)])

    ranks = np.zeros(5 * num_audios)
    top1 = np.zeros(5 * num_audios)

    for index in range(num_audios):

        # get query captions
        queries = cap_embs[5 * index: 5 * index + 5]

        # compute scores
        d = cosine_similarity(queries, audios).numpy()

        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    mAP10 = 100.0 * np.sum(1 / (ranks[np.where(ranks < 10)[0]] + 1)) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return r1, r5, r10, r50, medr, meanr, ranks, top1, mAP10
    else:
        return r1, r5, r10, r50, medr, meanr, mAP10


if __name__ == "__main__":
    import os
    import pandas as pd
    from src import laion_clap

    models = [
        "base"
        # "reweighting_5_1e3_acaps_clotho",
        # "reweighting_5_1e4_acaps_clotho",
        # "reweighting_7_1e2_acaps_clotho",
        # "reweighting_7_1e2_acaps_clotho_30",
        # "reweighting_7_1e3_acaps_clotho",
        # "reweighting_7_1e3_acaps_clotho_30",
        # "reweighting_7_1e4_acaps_clotho",
        # "reweighting_9_1e3_acaps_clotho",
        # "reweighting_9_1e4_acaps_clotho"
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
        # Initialize model
        model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
        # model.load_ckpt(f'logs/{model_name}/checkpoints/epoch_latest.pt', strict=False)
        model.load_ckpt("models/630k-audioset-best.pt", strict=False)

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
        text_embed = torch.nan_to_num(text_embed, nan=0.0).cpu()
        audio_embed = torch.nan_to_num(audio_embed, nan=0.0).cpu()

        # print(t2a(audio_embed, text_embed))

        print(a2t(audio_embed, text_embed))
