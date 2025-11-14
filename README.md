# smc-project
This project involves sound event detection and model fusion (PANNs / VGGish / AST).

Due to Canvas file size limit, data files are not included in the code submission. However, it can be downloaded via ```scripts/download_audioset_wavs.py```.

## environment setup
```bash
conda create -n sound_env python=3.9 -y
conda activate sound_env
pip install -r requirements.txt
```

## pretrained model download
```bash
python scripts/download_models.py
```

## Audioset download
```bash
python scripts/download_audioset_wavs.py
```

## fine-tuning
```bash
python scripts/finetune_panns.py
python scripts/finetune_vggish.py
```

## demo
```bash
streamlit run app.py
```


