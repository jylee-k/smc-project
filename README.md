# SilentSignals (smc-project)


SilentSignals is a real-time sound event detection prototype built for the deaf and hard-of-hearing. It uses a "mixture of experts" (MoE) model to listen to the environment and send tiered alerts for critical sounds.

**Novelty**: This project fuses the outputs of three distinct models (PANNs, VGGish, and AST) and uses custom label sets to "mask" away everyday, non-critical sounds to reduce notification fatigue.

## 🚨 Features

* **Real-Time Audio Processing**: Listens from the microphone in 2-second chunks.
* **Mixture of Experts (MoE)**: Fuses predictions from PANN, VGGish, and AST models for higher accuracy.
* **Tiered Alert System**: Classifies sounds into Critical, Warning, and Info tiers.
* **Custom Profiles**: Supports "Normal," "Sleep," and "DND" profiles to filter alerts.
* **Demo Mode**: Allows testing with uploaded audio files.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/smc-project.git](https://github.com/your-username/smc-project.git)
    cd smc-project
    ```

2.  **Set up the environment** (Python 3.11 is recommended):
    ```bash
    conda create -n sound_env python=3.11 -y
    conda activate sound_env
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Pre-trained Models:**
    This is a critical step. The script downloads the finetuned models required for the app.
    ```bash
    python scripts/download_models.py
    ```

## 🚀 Running the Demo

Once installed, run the Streamlit application:

```bash
streamlit run app.py