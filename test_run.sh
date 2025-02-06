
srun --pty --partition=cbcb-heng --account=cbcb-heng --qos=high \
    --mem-per-cpu=64G --gres=gpu  --time=12:00:00 \
    /fs/nexus-scratch/milis/miniconda3/clap/bin/python /fs/nexus-scratch/milis/848K/CLAP/src/laion_clap/training/my_dataloader.py
