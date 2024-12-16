import os
import pandas as pd
import torch
import json
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn.functional as F


def get_clotho(directory):
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

    print(len(data))
    return pd.DataFrame(data)


class AudioDataset(Dataset):
    def __init__(self, split, sample_rate=16000):
        # AudioCaps
        csv_file = f"/fs/nexus-scratch/milis/848K/CLAP/data/audiocaps/{split}.csv"
        audiocaps_dir = f"/fs/nexus-scratch/milis/848K/CLAP/data/audiocaps/audio_files/{split}"

        self.max_length = 10 * sample_rate
        # Get list of audio files in the directory
        audio_files = [file for file in os.listdir(audiocaps_dir) if file.endswith('.wav')]
        audio_ids = [os.path.basename(name).rsplit("_")[0] for name in audio_files]
        
        # Load and filter the CSV data from AudioCaps
        audiocaps_data = pd.read_csv(csv_file)
        audiocaps_data = audiocaps_data[
            audiocaps_data['youtube_id'].isin(audio_ids)
        ].reset_index(drop=True)

        if split == "train":
            # Clotho
            clotho_dir = f"/fs/nexus-scratch/milis/848K/CLAP/data/clotho_dataset/{split}"
            clotho_df = get_clotho(clotho_dir)

            self.data = pd.concat([audiocaps_data, clotho_df], ignore_index=True)
        else:
            self.data = audiocaps_data

        print("Data samples:", len(self.data))

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
            audio_name = f"{row['youtube_id']}_{row['start_time']}.wav"
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
        pass
