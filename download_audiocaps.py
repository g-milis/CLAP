import pandas as pd
import os
import requests
from moviepy.editor import AudioFileClip
from googleapiclient.discovery import build


# Use with all three splits
splits = ["train", "val", "test"]

splits = ["test"]

for split in splits:
    # Load dataset
    dataset_path = f'./data/audiocaps/{split}.csv'  # Update with your path to CSV
    data = pd.read_csv(dataset_path)

    # Create a directory to store audio files
    audio_dir = f'./data/audiocaps/audio_files/{split}'
    os.makedirs(audio_dir, exist_ok=True)

    # YouTube Data API setup
    API_KEY = "AIzaSyCE_6vZUi96EVubwDX5YJVmKAFvnJChUHM"  # Replace with your actual API key
    youtube = build("youtube", "v3", developerKey=API_KEY)

    def get_video_url(youtube_id):
        """Retrieve the YouTube video URL using the API."""
        try:
            # Fetch video details to validate existence
            request = youtube.videos().list(part="id", id=youtube_id)
            response = request.execute()
            if response["items"]:
                return f"https://www.youtube.com/watch?v={youtube_id}"
            else:
                print(f"Video with ID {youtube_id} not found.")
                return None
        except Exception as e:
            print(f"Error fetching video URL for {youtube_id}: {e}")
            return None

    def download_audio(youtube_url, youtube_id, start_time, duration=10, output_dir=audio_dir):
        """Download and trim audio using an alternative downloader."""
        start_time_sec = int(start_time)  # Convert start time to seconds
        trimmed_audio_path = os.path.join(output_dir, f"{youtube_id}_{start_time_sec}.wav")

        if os.path.exists(trimmed_audio_path):
            print("File exists, continuing")
            return

        try:
            # Use an external tool like `yt-dlp` to download the audio
            output_path = os.path.join(output_dir, f"{youtube_id}.mp4")
            os.system(f"yt-dlp -x --audio-format mp3 {youtube_url} -o {output_path}")

            # Load and trim audio using moviepy
            with AudioFileClip(output_path) as audio:
                audio.subclip(start_time_sec, start_time_sec + duration).write_audiofile(trimmed_audio_path)

        except Exception as e:
            print(f"Error processing {youtube_url}: {e}")

        # Always delete the mp4 file
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    # Download and trim all audio clips
    for _, row in data.iterrows():
        youtube_id = row['youtube_id']
        start_time = row['start_time']
        youtube_url = get_video_url(youtube_id)
        if youtube_url:
            download_audio(youtube_url, youtube_id, start_time)

    # captions = data['caption'].tolist()
