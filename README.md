# smc-project
## environment setup
conda create -n sound_env python=3.9 -y
conda activate sound_env
pip install -r requirements.txt

## pretrained model download
python scripts/download_models.py

## fine-tuning
python scripts/finetune_panns.py
python scripts/finetune_vggish.py

## demo
streamlit run app.py