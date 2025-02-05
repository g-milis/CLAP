# ONLY CHANGE THESE TWO PARAMETERS
# Use a different name each time, and use reweighting_level=-1 for full finetuning
reweighting_level=7
name=test_wavcaps


srun --pty --partition=cbcb-heng --account=cbcb-heng --qos=high \
    --mem-per-cpu=64G --gres=gpu  --time=12:00:00 \
    ../../miniconda3/clap/bin/python -m laion_clap.training.main \
        --save-frequency 6 \
        --save-most-recent \
        --datasetpath="data/audiocaps/audio_files/train" \
        --precision="fp32" \
        --batch-size=96 \
        --lr=1e-5 \
        --wd=0.0 \
        --epochs=20 \
        --use-bn-sync \
        --tmodel roberta \
        --amodel HTSAT-tiny \
        --resume models/630k-audioset-best.pt \
        --datasetnames "audiocaps" \
        --top-k-checkpoint-select-dataset="AudioCaps-test" \
        --top-k-checkpoint-select-metric="mAP@10" \
        --logs 'logs' \
        --gather-with-grad \
        --optimizer "adam" \
        --data-filling "repeatpad" \
        --data-truncating "rand_trunc" \
        --prefetch-factor 2 \
        --report-to "wandb" \
        --name $name \
        --wandb-notes $name \
        --reweighting_level $reweighting_level
