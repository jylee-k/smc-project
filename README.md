# CS5647 Sound & Music Computing Group Project
## SilentSignals: Real-time Event-based Alerts

## Overview

SilentSignals is a real-time sound event detection prototype. It is designed to assist the deaf and hard-of-hearing by capturing environmental sounds, processing them with a machine learning pipeline, and delivering tiered alerts based on the sound's classification.

The core of this project is a 'mixture of experts' (MoE) fusion model. It processes audio in real-time by combining the predictions from three distinct models: PANN, VGGish, and AST. The fusion weights and other parameters are managed in the `config.yaml` file. In this project, we try to address gaps highlighted by users via newly engineered features as mentioned in our presentation and report.

## Core Features

* **Real-time Audio Processing**: Listens from the microphone and processes audio in continuous chunks.
* **Tiered Alert System**: Classifies sounds into Critical, Warning, and Info categories, as defined in `label_tiers.json`.
* **Mixture of Experts (MoE) Fusion**: Combines predictions from PANN, VGGish, and AST models for more robust detection.
* **Customizable Profiles**: Includes 'Normal', 'Sleep', and 'DND' modes to filter alert sensitivity based on user preference, and allow for confidencethreshold tuning.
* **Demo Mode**: Allows users to analyze and test pre-recorded audio files via a file uploader.
* **Finetuning Submission**: Provides a UI to upload new audio clips to help improve user's quality of life.

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
    conda create -n sound_env python=3.11 -y
    conda activate sound_env
    ```

2.  **Install dependencies from the requirements file:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Required Step: Download Fine-tuned Models (from our google drive)

The application requires our fine-tuned model files to run. Run the following script to download and place them in the correct directory (they should be in the correct directory if you use the script).

```bash
python scripts/download_models.py
```

Running the Application
After installation and model download, run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser. You can then start the microphone or upload a demo audio file.

---

Model Finetuning (For Reference):
The scripts used to finetune the PANN and VGGish models are included in the scripts/ directory to demonstrate the model training process.

Running these scripts is not required to use the application.

The relevant files are:

1. scripts/finetune_panns.py

2. scripts/finetune_vggish.py

3. scripts/download_audioset_wavs.py (Used to get a subset of the training data)