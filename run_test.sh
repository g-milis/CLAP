srun --partition=cbcb-heng --account=cbcb-heng --qos=high --mem-per-cpu=128G --gres=gpu:rtx6000ada --time=1:00:00 ../../miniconda3/clap/bin/python test_text2audio.py >> results/new_text2audio.txt

# srun --partition=cbcb-heng --account=cbcb-heng --qos=medium --mem-per-cpu=32G --gres=gpu  --time=1:00:00 ../../miniconda3/clap/bin/python esc50_api.py >> esc50.txt

# srun --partition=cbcb-heng --account=cbcb-heng --qos=medium --mem-per-cpu=32G --gres=gpu  --time=1:00:00 ../../miniconda3/clap/bin/python test_audio2text.py >> test_audio2text.txt
