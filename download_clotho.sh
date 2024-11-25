# Define the correct URLs
audio_dev_url='https://zenodo.org/records/3490684/files/clotho_audio_development.7z?download=1'
captions_dev_url='https://zenodo.org/records/3490684/files/clotho_captions_development.csv?download=1'
audio_eval_url='https://zenodo.org/records/3490684/files/clotho_audio_evaluation.7z?download=1'
captions_eval_url='https://zenodo.org/records/3490684/files/clotho_captions_evaluation.csv?download=1'

# Download directory
download_dir='data/clotho_dataset'
mkdir -p data
mkdir -p $download_dir

# # Download files using wget
# wget -O $download_dir/clotho_audio_development.7z $audio_dev_url
# wget -O $download_dir/clotho_captions_development.csv $captions_dev_url
# wget -O $download_dir/clotho_audio_evaluation.7z $audio_eval_url
# wget -O $download_dir/clotho_captions_evaluation.csv $captions_eval_url


# Extract .7z files
py7zr x $download_dir/clotho_audio_development.7z $download_dir
py7zr x $download_dir/clotho_audio_evaluation.7z $download_dir
