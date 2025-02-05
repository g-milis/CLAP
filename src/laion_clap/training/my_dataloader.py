import os
import pandas as pd
import torch
import json
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn.functional as F


def get_clotho():
    directory = f"/fs/nexus-scratch/milis/848K/CLAP/data/clotho_dataset/{split}"
    data = []

    # Iterate through the directory
    for file in os.listdir(directory):
        if file.endswith(".json"):
            # Get the base name without extension
            basename = os.path.splitext(file)[0]
            
            # Corresponding .flac file
            flac_path = os.path.join(directory, f"{basename}.flac")
            
            # JSON file path
            json_path = os.path.join(directory, file)
            
            # Check if the corresponding .flac file exists
            if os.path.exists(flac_path):
                # Load captions from JSON
                with open(json_path) as f:
                    captions = json.load(f)["text"]
                
                # Add entries to the data list
                for caption in captions:
                    data.append({
                        "audiocap_id": flac_path,
                        "youtube_id": flac_path,
                        "start_time": 0,
                        "caption": caption
                    })

    return pd.DataFrame(data)


def get_clotho_new():
    clotho_dir = "/fs/nexus-scratch/milis/848K/CLAP/data/clotho_dataset/development"
    csv_file = "/fs/nexus-scratch/milis/848K/CLAP/data/clotho_dataset/clotho_captions_development.csv"
    
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
    # # Iterate through the directory
    # for file in os.listdir(directory):
    #     if file.endswith(".json"):
    # json_path = os.path.join(directory, file)

    # Get blacklisted files to avoid test set leakage
    blacklisted_ids = []
    for blacklist_path in [
        "/fs/nexus-scratch/milis/848K/CLAP/WavCaps/data/json_files/blacklist/blacklist_exclude_test_ac.json",
        "/fs/nexus-scratch/milis/848K/CLAP/WavCaps/data/json_files/blacklist/blacklist_exclude_ub8k_esc50_vggsound.json"
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
        else:
            print(f"Audio file not found: {audio_path}")
    # print("Total WavCaps samples:", len(data))
    return pd.DataFrame(data)


class AudioDataset(Dataset):
    def __init__(self, split, sample_rate=16000):
        # AudioCaps
        csv_file = f"/fs/nexus-scratch/milis/848K/CLAP/data/audiocaps/{split}.csv"
        audiocaps_dir = f"/fs/nexus-scratch/milis/848K/CLAP/AudioCaps_CVSSP/waveforms/{split}"

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
                directory="/fs/cbcb-scratch/milis/data/wavcaps/AudioSet_SL_flac/",
                json_file="/fs/nexus-scratch/milis/848K/CLAP/WavCaps/data/json_files/AudioSet_SL/as_final.json",
                replace_wav=True
            )
            self.data = pd.concat([self.data, wavecaps_df_AudioSet], ignore_index=True)
            print("AudioSet samples:", len(wavecaps_df_AudioSet))

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

        waveform, orig_sample_rate = torchaudio.load(audio_path)

        # Resample if necessary
        if orig_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sample_rate, self.sample_rate)
            waveform = resampler(waveform)

        num_samples = waveform.size(1)  # Shape: (channels, samples)
        target_length = self.max_length or num_samples  # Use the current waveform length if max_length is not set

        # Zero-pad the waveform if it's shorter than the target length
        if num_samples < target_length:
            padding = target_length - num_samples
            waveform = F.pad(waveform, (0, padding))  # Pad on the right
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
    

def collate_fn(batch):
    """Custom collate function to handle missing files."""
    batch = [item for item in batch if item is not None]  # Remove None entries
    return batch



# Example usage
if __name__ == "__main__":
    # Define the dataset
    split = "train"
    dataset = AudioDataset(
        split
    )

    # Wrap the dataset in a DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Iterate through the data
    for batch in dataloader:
        print(batch)
        break
