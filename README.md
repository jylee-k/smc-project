# SilentSignals (smc-project)

## Overview

SilentSignals is a real-time sound event detection prototype. It is designed to assist the deaf and hard-of-hearing by capturing environmental sounds, processing them with a machine learning pipeline, and delivering tiered alerts based on the sound's classification.

The core of this project is a 'mixture of experts' (MoE) fusion model. It processes audio in real-time by combining the predictions from three distinct models: PANN, VGGish, and AST. The fusion weights and other parameters are managed in the `config.yaml` file.

## Core Features

* **Real-time Audio Processing**: Listens from the microphone and processes audio in continuous chunks.
* **Tiered Alert System**: Classifies sounds into Critical, Warning, and Info categories, as defined in `label_tiers.json`.
* **Mixture of Experts (MoE) Fusion**: Combines predictions from PANN, VGGish, and AST models for more robust detection.
* **Customizable Profiles**: Includes 'Normal', 'Sleep', and 'DND' modes to filter alert sensitivity based on user preference.
* **Demo Mode**: Allows users to analyze and test pre-recorded audio files via a file uploader.
* **Finetuning Submission**: Provides a UI to upload new audio clips to help improve the model's accuracy.

## Installation

You can install the environment using **either** `uv` (recommended) or `conda`.

1.  **Unzip the project folder** and navigate into it using your terminal.

---

### Option 1: Using `uv` (Fast)

1.  **Create and activate a virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```

2.  **Sync the locked dependencies:**
    ```bash
    uv sync
    ```

---

### Option 2: Using `conda` + `pip`

1.  **Create and activate the conda environment:**
    ```bash
    conda create -n sound_env python=3.9 -y
    conda activate sound_env
    ```

2.  **Install dependencies from the requirements file:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Required Step: Download Pre-trained Models

The application requires pre-trained model files to run. Run the following script to download and place them in the correct directory.

```bash
python scripts/download_models.py

Running the Application
After installation and model download, run the Streamlit application:

Bash

streamlit run app.py
The application will open in your browser. You can then start the microphone or upload a demo audio file.

Model Finetuning (For Reference)
The scripts used to finetune the PANN and VGGish models are included in the scripts/ directory to demonstrate the model training process.

Running these scripts is not required to use the application.

The relevant files are:

scripts/finetune_panns.py

scripts/finetune_vggish.py

scripts/download_audioset_wavs.py (Used to get the training data)

Project Structure
├── app.py                  # Main Streamlit application
├── realtime_solo.py        # Core MoE pipeline logic
├── record_sound.py         # Utility for real-time audio chunking
├── config.yaml             # Configuration for models, weights, and logs
├── label_tiers.json        # Defines alert tiers for sound labels
├── requirements.txt        # Dependencies for pip
├── pyproject.toml          # Dependencies for uv
├── uv.lock                 # Locked dependencies for uv
├── ast/                    # Source code for the AST model
│   └── src/models/ast_models.py
│   └── egs/audioset/class_labels_indices.csv
├── data/                   # Data files for labels and training manifests
│   ├── custom_label.json
│   └── balanced_train_segments.csv
├── scripts/                # Helper scripts
│   ├── download_models.py
│   ├── download_audioset_wavs.py
│   ├── finetune_panns.py
│   └── finetune_vggish.py
├── pretrained_model/       # (Downloaded) Contains the .pth/.pt model files
└── runs/                   # (Generated) Default output directory for prediction logs