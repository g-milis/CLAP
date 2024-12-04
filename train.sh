srun --pty --partition=cbcb-heng --account=cbcb-heng --qos=medium \
    --mem-per-cpu=64G --gres=gpu  --time=1:00:00 \
    ../../miniconda3/clap/bin/python -m laion_clap.training.main \
        --save-frequency 1 \
        --save-top-performance 3 \
        --save-most-recent \
        --datasetpath="data/audiocaps/audio_files/train" \
        --precision="fp32" \
        --batch-size=96 \
        --lr=1e-5 \
        --wd=0.0 \
        --epochs=18 \
        --use-bn-sync \
        --tmodel roberta \
        --amodel HTSAT-tiny \
        --resume models/630k-audioset-best.pt \
        --warmup 3200 \
        --datasetnames "AudioCaps" \
        --datasetinfos "train" \
        --top-k-checkpoint-select-dataset="AudioCaps-test" \
        --top-k-checkpoint-select-metric="mAP@10" \
        --logs 'logs' \
        --seed 3407 \
        --gather-with-grad \
        --optimizer "adam" \
        --data-filling "repeatpad" \
        --data-truncating "rand_trunc" \
        --prefetch-factor 2
        # --report-to "tensorboard" \
        # --tensorboard_path log/test_run



        # --pretrained models/630k-audioset-best.pt \