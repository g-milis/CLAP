import os
import torchaudio
import torch.nn.functional as F

torchaudio.set_audio_backend("sox_io")


sample_rate = 16000
max_length = 10 * sample_rate

audio_dirs = [
    "/fs/cbcb-scratch/milis/data/wavcaps/BBC_Sound_Effects_flac",
    "/fs/cbcb-scratch/milis/data/wavcaps/AudioSet_SL_flac",
    "/vulcanscratch/simin95/CLAP/data/wavcaps/SoundBible",
    "/vulcanscratch/simin95/CLAP/data/wavcaps/FreeSound_new"
]

for audio_dir in audio_dirs:
    print(audio_dir)

    for i, audio_path in enumerate(os.listdir(audio_dir)):
        if i == 5: break

        audio_path = os.path.join(audio_dir, audio_path)

        waveform, orig_sample_rate = torchaudio.load(audio_path)
        print(orig_sample_rate)

        # Resample if necessary
        if orig_sample_rate != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sample_rate, sample_rate)
            waveform = resampler(waveform)

        num_samples = waveform.size(1)  # Shape: (channels, samples)
        target_length = max_length or num_samples  # Use the current waveform length if max_length is not set

        # Zero-pad the waveform if it's shorter than the target length
        if num_samples < target_length:
            padding = target_length - num_samples
            waveform = F.pad(waveform, (0, padding))  # Pad on the right
        elif num_samples > target_length:
            waveform = waveform[..., :target_length]

        waveform = waveform.mean(dim=0)
