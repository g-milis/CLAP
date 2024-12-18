from src import laion_clap
import glob
import json
import torch
import numpy as np

device = torch.device('cuda:0')

# download https://drive.google.com/drive/folders/1scyH43eQAcrBz-5fAw44C6RNBhC3ejvX?usp=sharing and extract ./ESC50_1/test/0.tar to ./ESC50_1/test/
esc50_test_dir = 'data/ESC50_1/test'
class_index_dict_path = 'class_labels/ESC50_class_labels_indices_space.json'

models = [
    "base"
    "reweighting_5_1e3_acaps_clotho",
    "reweighting_5_1e4_acaps_clotho",
    "reweighting_7_1e2_acaps_clotho",
    "reweighting_7_1e2_acaps_clotho_30",
    "reweighting_7_1e3_acaps_clotho",
    "reweighting_7_1e3_acaps_clotho_30",
    "reweighting_7_1e4_acaps_clotho",
    "reweighting_9_1e3_acaps_clotho",
    "reweighting_9_1e4_acaps_clotho",
    "reweighting_5_1e5_acaps_clotho",
    "reweighting_7_1e5_acaps_clotho",
    "reweighting_9_1e5_acaps_clotho",
]


for model_name in models:
    # Load the model
    model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
    if model_name == "base":
        model.load_ckpt('models/630k-audioset-best.pt', strict=False)
    else:
        model.load_ckpt(f'logs/{model_name}/checkpoints/epoch_latest.pt', strict=False)

    # Get the class index dict
    class_index_dict = {v: k for v, k in json.load(open(class_index_dict_path)).items()}

    # Get all the data
    audio_files = sorted(glob.glob(esc50_test_dir + '**/*.flac', recursive=True))
    json_files = sorted(glob.glob(esc50_test_dir + '**/*.json', recursive=True))
    ground_truth_idx = [class_index_dict[json.load(open(jf))['tag'][0]] for jf in json_files]

    with torch.no_grad():
        ground_truth = torch.tensor(ground_truth_idx).view(-1, 1)

        # Get text features
        all_texts = ["This is a sound of " + t for t in class_index_dict.keys()]
        text_embed = model.get_text_embedding(all_texts)
        audio_embed = model.get_audio_embedding_from_filelist(x=audio_files)

        ranking = torch.argsort(torch.tensor(audio_embed) @ torch.tensor(text_embed).t(), descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.cpu().numpy()

        metrics = {}
        metrics[f"mean_rank"] = preds.mean() + 1
        metrics[f"median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"R@{k}"] = np.mean(preds < k)
        # map@10
        metrics[f"mAP@10"] = np.mean(np.where(preds < 10, 1 / (preds + 1), 0.0))

        print(
            f"Zeroshot Classification Results: "
            + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
        )
