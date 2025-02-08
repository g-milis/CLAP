# ONLY CHANGE THESE TWO PARAMETERS
# Use a different name each time, and use reweighting_level=-1 for full finetuning
# Use numbers from 1 to 12
# ALSO CHANGE YOUR PARTITION AND ACCOUNT!
reweighting_level=5
lr=1e-4
name=reweighting_5_1e4

# No need to change these
batch_size=64
threads=4

srun --pty --partition=cbcb-heng --account=cbcb-heng --qos=high \
    --mem-per-cpu=64G --gres=gpu --time=18:00:00 \
    ../../miniconda3/clap/bin/python -m laion_clap.training.main \
        --save-frequency 6 \
        --save-most-recent \
        --precision="fp32" \
        --batch-size=$batch_size \
        --lr=$lr \
        --wd=0.0 \
        --epochs=20 \
        --workers=$threads  \
        --use-bn-sync \
        --tmodel roberta \
        --amodel HTSAT-tiny \
        --resume models/630k-audioset-best.pt \
        --logs 'logs' \
        --gather-with-grad \
        --optimizer "adam" \
        --data-filling "repeatpad" \
        --data-truncating "rand_trunc" \
        --report-to "wandb" \
        --name $name \
        --wandb-notes $name \
        --reweighting_level $reweighting_level
