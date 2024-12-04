import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.transforms import MelSpectrogram
import torch.nn.functional as F


class AudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, transform=None, sample_rate=16000, mel_spec_params=None):
        """
        Args:
            csv_file (str): Path to the CSV file with captions and audio file paths.
            audio_dir (str): Directory where the audio files are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
            sample_rate (int): Target sample rate for audio files.
            mel_spec_params (dict): Parameters for MelSpectrogram, if mel spectrogram is needed.
        """
        self.max_length = 10 * sample_rate
        # Get list of audio files in the directory
        audio_files = [file for file in os.listdir(audio_dir) if file.endswith('.wav')]
        audio_ids = [os.path.basename(name).rsplit("_")[0] for name in audio_files]
        
        # Load and filter the CSV data
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['youtube_id'].isin(audio_ids)].reset_index(drop=True)

        # Raise an error if no valid data is found
        if self.data.empty:
            raise ValueError("No matching audio files found in the directory for the provided captions.")

        # Store directory and parameters
        self.audio_dir = audio_dir
        self.transform = transform
        self.sample_rate = sample_rate
        self.mel_transform = MelSpectrogram(**mel_spec_params) if mel_spec_params else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract row information
        row = self.data.iloc[idx]
        audio_name = row['youtube_id']
        text = row['caption']
        # target = row['class_label']  # Assuming labels are stored under `class_label`
        # longer = row.get('longer', False)  # Optional field

        audio_name = f"{row['youtube_id']}_{row['start_time']}.wav"

        # Construct full path
        audio_path = os.path.join(self.audio_dir, audio_name)

        # print(audio_path, text)

        waveform, orig_sample_rate = torchaudio.load(audio_path)

        # print(waveform.shape, audio_path)

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

        waveform = waveform.mean(dim=0)
        # print(waveform.shape)

        # Apply mel spectrogram transform if specified
        # mel_spec = self.mel_transform(waveform) if self.mel_transform else None

        # Apply additional transformation
        if self.transform:
            waveform = self.transform(waveform)

        # Construct data dictionary
        data_dict = {
            #"hdf5_path": None,  # Placeholder for HDF5 path if needed
            "index_in_hdf5": idx,
            "audio_name": audio_name,
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
    dataset = AudioDataset(
        csv_file="/fs/nexus-scratch/milis/848K/CLAP/data/audiocaps/train.csv",
        audio_dir="/fs/nexus-scratch/milis/848K/CLAP/data/audiocaps/audio_files/train",
        # mel_spec_params={"sample_rate": 16000, "n_fft": 1024, "hop_length": 512, "n_mels": 64}
    )

    print(len(dataset))

    # Wrap the dataset in a DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)#, collate_fn=collate_fn)

    # Iterate through the data
    for batch in dataloader:
        pass