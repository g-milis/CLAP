import os
import requests
from tqdm import tqdm
import shutil


# Base URL for the repository
base_url = "https://huggingface.co/datasets/cvssp/WavCaps/resolve/main/Zip_files/BBC_Sound_Effects/"

# List of split archive files
zip_files = [
    "BBC_Sound_Effects.z01",
    "BBC_Sound_Effects.z02",
    "BBC_Sound_Effects.z03",
    "BBC_Sound_Effects.z04",
    "BBC_Sound_Effects.z05",
    "BBC_Sound_Effects.z06",
    "BBC_Sound_Effects.z07",
    "BBC_Sound_Effects.z08",
    "BBC_Sound_Effects.z09",
    "BBC_Sound_Effects.z10",
    "BBC_Sound_Effects.z11",
    "BBC_Sound_Effects.z12",
    "BBC_Sound_Effects.z13",
    "BBC_Sound_Effects.z14",
    "BBC_Sound_Effects.z15",
    "BBC_Sound_Effects.z16",
    "BBC_Sound_Effects.z17",
    "BBC_Sound_Effects.z18",
    "BBC_Sound_Effects.z19",
    "BBC_Sound_Effects.z20",
    "BBC_Sound_Effects.z21",
    "BBC_Sound_Effects.z22",
    "BBC_Sound_Effects.z23",
    "BBC_Sound_Effects.z24",
    "BBC_Sound_Effects.zip",
]

# Directory to save the downloaded files and extract them
# Use absolute paths to avoid issues with relative paths
download_dir = "/fs/cbcb-scratch/milis/data/wavcaps/download"  # Change this to a valid path in your environment
extract_dir = "/fs/cbcb-scratch/milis/data/wavcaps/BBC_Sound_Effects"  # Change this to a valid path in your environment

# Create directories if they don't exist
os.makedirs(download_dir, exist_ok=True)
os.makedirs(extract_dir, exist_ok=True)

# Function to download a file
def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(dest_path, "wb") as f, tqdm(
        desc=f"Downloading {os.path.basename(dest_path)}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

# # Download all the split archive files
# for zip_file in zip_files:
#     zip_url = base_url + zip_file
#     local_zip_path = os.path.join(download_dir, zip_file)
#     # Download the split archive file
#     download_file(zip_url, local_zip_path)

# Combine the split archive parts using 7z command
combined_zip_path = os.path.join(download_dir, "BBC_Sound_Effects_combined.zip")
# Change working directory to the download directory
os.chdir(download_dir)

# Use 7z to extract the archive parts
# If you don't have `7z` installed in your environment, this would need to be installed first
os.system(f"7za x {zip_files[-1]}")  # Extract the last zip part which contains the metadata for the split archive

# Check if extraction was successful
if os.path.exists(combined_zip_path):
    # Now, unzip the final archive (if not done by 7z already)
    os.system(f"unzip {combined_zip_path} -d {extract_dir}")
    # Change working directory back to the original
    os.chdir("..")
    print(f"Extracted contents to: {extract_dir}")
    # Cleanup: Delete split archive files and the combined archive
    for part_file in zip_files:
        os.remove(os.path.join(download_dir, part_file))
    print("Cleanup completed. Only extracted content remains.")
else:
    print(f"Error: {combined_zip_path} was not created successfully.")
