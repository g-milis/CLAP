srun --pty --partition=cbcb-heng --account=cbcb-heng --qos=medium \
    --mem-per-cpu=64G --gres=gpu  --time=1:00:00 \
    ../../miniconda3/clap/bin/python simin_test.py
