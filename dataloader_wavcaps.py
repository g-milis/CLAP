import os
import pandas as pd
import torch
import json
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn.functional as F


def get_wavecaps(directory, json_file):
    """Load metadata and captions from WavCaps dataset."""
    data = []
    # # Iterate through the directory
    # for file in os.listdir(directory):
    #     if file.endswith(".json"):
    # json_path = os.path.join(directory, file)
    # Load the JSON file
    if not os.path.exists(json_file):
        print(f"JSON file not found: {json_file}")
        return
    with open(json_file, "r") as f:
        json_data = json.load(f)
    # Debug: Check the keys in the JSON structure
    print(f"JSON keys: {json_data.keys()}")
    # Iterate through the data list in the JSON
    for entry in json_data.get("data", []):
        audio_id = entry["id"]  # Extract audio file ID (with .wav extension)
        # Construct the full path to the audio file
        audio_path = os.path.join(directory, audio_id) + ".flac"
        print(f"Audio path: {audio_path}")
        # Check if the audio file exists
        if os.path.exists(audio_path):
            data.append({
                "audio_path": audio_path,
                "caption": entry["caption"],
                "duration": entry["duration"],
            })
        else:
            print(f"Audio file not found: {audio_path}")
    print("Total WavCaps samples:", len(data))
    return pd.DataFrame(data)


# Updated WavCaps-only AudioDataset
class AudioDataset(Dataset):
    """Dataset class for loading WavCaps data."""
    def __init__(self, sample_rate=16000):
        # Path to the WavCaps data
        base_dir = "/fs/cbcb-scratch/milis/data/wavcaps/download/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/BBC_Sound_Effects_flac"
        wavecaps_dir = base_dir

        json_file = "/fs/nexus-scratch/milis/848K/CLAP/WavCaps/data/json_files/BBC_Sound_Effects/bbc_final.json"
        # Load WavCaps metadata
        wavecaps_df = get_wavecaps(wavecaps_dir, json_file)
        # Store data
        self.data = wavecaps_df
        print("Total WavCaps data samples:", len(self.data))
        # Store directory and parameters
        self.sample_rate = sample_rate
        self.max_length = 10 * sample_rate


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract row information
        row = self.data.iloc[idx]
        audio_path = row['audio_path']
        text = row['caption']
        # Load the audio file
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
            "index_in_hdf5": idx,
            "audio_name": audio_path,
            "waveform": waveform,
            "text": text,
        }
        return data_dict


if __name__ == "__main__":
    # Define the dataset
    dataset = AudioDataset(sample_rate=16000)
    # Wrap the dataset in a DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # Iterate through the data
    for batch in dataloader:
        print(batch)  # Display batch data
        break  # Exit after the first batch for quick testing
