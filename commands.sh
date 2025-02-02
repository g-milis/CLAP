pip --cache-dir=/fs/nexus-scratch/milis/cache install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

../../miniconda3/clap/bin/python download_audiocaps.py 

conda activate /fs/nexus-scratch/milis/miniconda3/clap


gdown --no-check-certificate --folder 1scyH43eQAcrBz-5fAw44C6RNBhC3ejvX
tar xvf ESC50_1/train/0.tar


# [milis@nexuscbcb01 ~]$ chown milis:clap /fs/cbcb-scratch/milis/data
# [milis@nexuscbcb01 ~]$ chown milis:clap /fs/cbcb-scratch/milis/data/wavcaps
