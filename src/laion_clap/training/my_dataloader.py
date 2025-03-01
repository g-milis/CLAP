import os
import pandas as pd
import torch
import json
import random
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.functional as F

torchaudio.set_audio_backend("sox_io")


CLOTHO_ROOT_PATH = "/fs/cbcb-scratch/milis/data/clotho_dataset"
AUDIOCAPS_CSVS_PATH = "/fs/cbcb-scratch/milis/data/audiocaps"
AUDIOCAPS_WAVEFORMS_PATH = "/fs/cbcb-scratch/milis/data/AudioCaps_CVSSP/waveforms"


# def get_clotho():
#     directory = f"/fs/cbcb-scratch/milis/data/clotho_dataset/{split}"
#     data = []

#     # Iterate through the directory
#     for file in os.listdir(directory):
#         if file.endswith(".json"):
#             # Get the base name without extension
#             basename = os.path.splitext(file)[0]
            
#             # Corresponding .flac file
#             flac_path = os.path.join(directory, f"{basename}.flac")
            
#             # JSON file path
#             json_path = os.path.join(directory, file)
            
#             # Check if the corresponding .flac file exists
#             if os.path.exists(flac_path):
#                 # Load captions from JSON
#                 with open(json_path) as f:
#                     captions = json.load(f)["text"]
                
#                 # Add entries to the data list
#                 for caption in captions:
#                     data.append({
#                         "audiocap_id": flac_path,
#                         "youtube_id": flac_path,
#                         "start_time": 0,
#                         "caption": caption
#                     })

#     return pd.DataFrame(data)


def get_clotho_new():
    clotho_dir = f"{CLOTHO_ROOT_PATH}/development"
    csv_file = f"{CLOTHO_ROOT_PATH}/clotho_captions_development.csv"
    
    data = []
    
    with open(csv_file, encoding="utf-8") as file:
        next(file)  # Skip the first line
        
        for line in file:
            parts = line.strip().split(",", maxsplit=1)
            if len(parts) < 2:
                continue
            file_name, captions = parts[0], parts[1]
            captions_list = captions.split(",")

            audio_path = os.path.join(clotho_dir, file_name)
            
            for caption in captions_list:
                if os.path.exists(audio_path):
                    data.append({
                        "audiocap_id": audio_path,
                        "youtube_id": audio_path,
                        "start_time": 0,
                        "caption": caption
                    })
    
    return pd.DataFrame(data)


def get_wavecaps_subset(directory="", json_file="", replace_wav=False):
    data = []

    # Get blacklisted files to avoid test set leakage
    blacklisted_ids = []
    for blacklist_path in [
        "WavCaps/data/json_files/blacklist/blacklist_exclude_test_ac.json",
        "WavCaps/data/json_files/blacklist/blacklist_exclude_ub8k_esc50_vggsound.json"
    ]:
        with open(blacklist_path) as f:
            blacklist_json = json.load(f)
        blacklisted_ids += blacklist_json["AudioSet"] + blacklist_json["FreeSound"]

    # Load the JSON file
    if not os.path.exists(json_file):
        print(f"JSON file not found: {json_file}")
        return
    with open(json_file, "r") as f:
        json_data = json.load(f)
    # Iterate through the data list in the JSON
    for entry in json_data.get("data", []):
        audio_id = entry["id"]  # Extract audio file ID (with .wav extension)

        if audio_id in blacklisted_ids:
            # print(audio_id)
            continue

        if replace_wav:
            audio_path = os.path.join(directory, audio_id).replace(".wav", ".flac")
        else:
            audio_path = os.path.join(directory, audio_id) + ".flac"
        # print(f"Audio path: {audio_path}")
        # Check if the audio file exists
        if os.path.exists(audio_path):
            data.append({
                "audiocap_id": audio_path,
                "youtube_id": audio_path,
                "start_time": 0,
                "caption": entry["caption"]
                #"duration": entry["duration"],
            })
    return pd.DataFrame(data)


class BigAudioDataset(Dataset):
    def __init__(self, split, sample_rate=16000):
        # AudioCaps
        csv_file = f"{AUDIOCAPS_CSVS_PATH}/{split}.csv"
        audiocaps_dir = f"{AUDIOCAPS_WAVEFORMS_PATH}/{split}"

        self.max_length = 10 * sample_rate
        # Get list of audio files in the directory
        audio_files = [file for file in os.listdir(audiocaps_dir) if file.endswith('.wav')]
        audio_ids = [os.path.basename(name).rsplit("_")[0][1:].replace(".wav", "") for name in audio_files]
        
        # Load and filter the CSV data from AudioCaps
        audiocaps_data = pd.read_csv(csv_file)
        audiocaps_data = audiocaps_data[
            audiocaps_data['youtube_id'].isin(audio_ids)
        ].reset_index(drop=True)

        self.data = audiocaps_data

        print("AudioCaps samples:", len(audiocaps_data))

        if split == "train":
            # Clotho
            clotho_df = get_clotho_new()
            self.data = pd.concat([self.data, clotho_df], ignore_index=True)
            print("Clotho samples:", len(clotho_df))

            # BBC_Sound_Effects
            wavecaps_df_BBC_Sound_Effects = get_wavecaps_subset(
                directory="/fs/cbcb-scratch/milis/data/wavcaps/BBC_Sound_Effects_flac",
                json_file="/fs/nexus-scratch/milis/848K/CLAP/WavCaps/data/json_files/BBC_Sound_Effects/bbc_final.json"
            )
            self.data = pd.concat([self.data, wavecaps_df_BBC_Sound_Effects], ignore_index=True)
            print("BBC_Sound_Effects samples:", len(wavecaps_df_BBC_Sound_Effects))

            # AudioSet
            wavecaps_df_AudioSet = get_wavecaps_subset(
                directory="/fs/cbcb-scratch/milis/data/wavcaps/AudioSet_SL_flac",
                json_file="/fs/nexus-scratch/milis/848K/CLAP/WavCaps/data/json_files/AudioSet_SL/as_final.json",
                replace_wav=True
            )
            self.data = pd.concat([self.data, wavecaps_df_AudioSet], ignore_index=True)
            print("AudioSet samples:", len(wavecaps_df_AudioSet))

            # SoundBible
            wavecaps_df_SoundBible = get_wavecaps_subset(
                directory="/vulcanscratch/simin95/CLAP/data/wavcaps/SoundBible",
                json_file="/fs/nexus-scratch/milis/848K/CLAP/WavCaps/data/json_files/SoundBible/sb_final.json"
            )
            self.data = pd.concat([self.data, wavecaps_df_SoundBible], ignore_index=True)
            print("SoundBible samples:", len(wavecaps_df_SoundBible))

            # FreeSound
            wavecaps_df_FreeSound = get_wavecaps_subset(
                directory="/vulcanscratch/simin95/CLAP/data/wavcaps/FreeSound_new",
                json_file="/fs/cbcb-scratch/milis/data/wavcaps/fsd_final.json"
            )
            self.data = pd.concat([self.data, wavecaps_df_FreeSound], ignore_index=True)
            print("FreeSound samples:", len(wavecaps_df_FreeSound))

        # Remove problematic paths, some are duplicate
        with open("problematic_files.txt") as f:
            problematic_paths = set([line.strip() for line in f.readlines()])

        self.data = self.data[
            ~self.data['youtube_id'].isin(problematic_paths)
        ].reset_index(drop=True)

        print("Total samples:", len(self.data))

        # Store directory and parameters
        self.audiocaps_dir = audiocaps_dir
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract row information
        row = self.data.iloc[idx]
        audio_path = row['youtube_id']
        text = row['caption']
        # target = row['class_label']  # Assuming labels are stored under `class_label`
        # longer = row.get('longer', False)  # Optional field

        if not audio_path.endswith(".flac"):
            if not audio_path.endswith(".wav"):
                audio_name = f"Y{row['youtube_id']}.wav"
                audio_path = os.path.join(self.audiocaps_dir, audio_name)
        
        try:
            # Get file metadata to determine total frames
            info = torchaudio.info(audio_path)
            total_samples = info.num_frames

            segment_samples = info.sample_rate * 10

            # Ensure we don't exceed the total file length
            if total_samples > segment_samples:
                start_sample = random.randint(0, total_samples - segment_samples)
            else:
                start_sample = 0  # If file is too short, take from the beginning

            # Load only the necessary part of the file
            waveform, orig_sample_rate = torchaudio.load(audio_path, frame_offset=start_sample, num_frames=segment_samples)

            # Resample if necessary
            if orig_sample_rate != self.sample_rate:
                waveform = F.resample(waveform, orig_freq=orig_sample_rate, new_freq=self.sample_rate)

        except RuntimeError:
            print("Problem:", audio_path)
            return None

        num_samples = waveform.size(1)  # Shape: (channels, samples)
        target_length = self.max_length or num_samples  # Use the current waveform length if max_length is not set

        # Zero-pad the waveform if it's shorter than the target length
        if num_samples < target_length:
            padding = target_length - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))  # Pad on the right
        elif num_samples > target_length:
            waveform = waveform[..., :target_length]

        waveform = waveform.mean(dim=0)

        # Construct data dictionary
        data_dict = {
            #"hdf5_path": None,  # Placeholder for HDF5 path if needed
            "index_in_hdf5": idx,
            "audio_name": audio_path,
            "waveform": waveform,
            #"class_label": target,
            "text": text,
            #"longer": longer,
            #"mel_fusion": mel_spec
        }

        return data_dict
    

    def collate_fn(self, batch):
        """Custom collate function to handle missing files."""
        batch = [item for item in batch if item is not None]
        return batch


# Example usage
if __name__ == "__main__":
    import sys
    from tqdm import tqdm

    # Define the dataset
    split = "train"
    dataset = BigAudioDataset(
        split
    )

    # Wrap the dataset in a DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )

    # Iterate through the data
    for batch in tqdm(dataloader, file=sys.stderr):
        pass
