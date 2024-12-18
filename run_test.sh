srun --pty --partition=cbcb-heng --account=cbcb-heng --qos=medium \
    --mem-per-cpu=64G --gres=gpu  --time=1:00:00 \
    ../../miniconda3/clap/bin/python test_text2audio.py  >> test_text2audio.txt
    
srun --pty --partition=cbcb-heng --account=cbcb-heng --qos=medium \
    --mem-per-cpu=64G --gres=gpu  --time=1:00:00 \
    esc50_api.py >> esc50.txt

srun --pty --partition=cbcb-heng --account=cbcb-heng --qos=medium \
    --mem-per-cpu=64G --gres=gpu  --time=1:00:00 \
    test_audio2text.py >> test_audio2text.txt
