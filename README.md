# Tomato Disease Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Enabled-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Inference-green)
![Status](https://img.shields.io/badge/Status-Active-success)

A comprehensive Deep Learning solution for detecting and classifying 10 different types of tomato plant diseases. This project utilizes **EfficientNetB3** (a state-of-the-art convolutional neural network) for high-accuracy image classification and serves the model via a modern **FastAPI** backend with a responsive web interface.

## Key Features

*   **High Accuracy**: Built on `EfficientNetB3`, leveraging transfer learning for robust performance.
*   **Real-time Inference**: Fast predictions using a lightweight FastAPI server.
*   **User-Friendly Interface**: Clean, modern web UI for easy image uploading and visualization.
*   **GPU Acceleration**: Fully optimized for NVIDIA GPUs (CUDA support) for rapid training.
*   **Automated Setup**: Includes scripts for one-click environment configuration.

## Dataset Classes

The model is trained to identify the following 10 classes:
*   Tomato Bacterial Spot
*   Tomato Early Blight
*   Tomato Late Blight
*   Tomato Leaf Mold
*   Tomato Septoria Leaf Spot
*   Tomato Spider Mites (Two-spotted spider mite)
*   Tomato Target Spot
*   Tomato Yellow Leaf Curl Virus
*   Tomato Mosaic Virus
*   Tomato Healthy

## Technical Stack

*   **Deep Learning**: PyTorch, Torchvision
*   **Backend**: FastAPI, Uvicorn
*   **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
*   **Data Processing**: Pillow (PIL), NumPy
*   **Environment**: Python Virtualenv

## Installation and Setup

### Prerequisites
*   **Python 3.10 or 3.11** (Required for GPU support)
*   **NVIDIA GPU** (Optional, but recommended for training)

### 1. Clone the Repository
```bash
git clone https://github.com/Prajwallnaik/plant_disease_model.git
cd tomato-disease-detector
```

### 2. Automatic Setup (Windows)
We provide a batch script to automatically create a virtual environment and install the correct dependencies (including CUDA for GPU).
Just double-click **`setup_gpu_env.bat`** or run:
```cmd
setup_gpu_env.bat
```

### 3. Manual Setup (Alternative)
If you prefer manual installation:
```bash
python -m venv venv
venv\Scripts\activate
# For GPU (CUDA 12.4):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# Other dependencies:
pip install -r requirements.txt
```

## Training the Model

To train the model from scratch on your dataset:

1.  Place your dataset in the `data/` directory with `train`, `val`, and `test` subfolders.
2.  Run the training script:
    ```bash
    venv\Scripts\activate
    python train.py
    ```
    *   *Note: This will save the best model to `model/efficientnet_b3_tomato.pth`.*

## Running the Application

Once the model is trained, you can start the prediction server:

1.  Start the FastAPI app:
    ```bash
    venv\Scripts\activate
    uvicorn app.main:app --reload
    ```
2.  Open your browser and visit:
    [http://localhost:8000](http://localhost:8000)

## Project Structure

```
tomato-disease-detector/
├── app/
│   ├── main.py              # FastAPI application & inference logic
│   ├── utils.py             # Utility functions
│   └── static/
│       └── index.html       # Web Frontend
├── data/                    # Dataset directory (train/val/test)
├── model/                   # Model storage
│   └── efficientnet_b3_tomato.pth  # Trained PyTorch model
├── train.py                 # Training script
├── split_data.py            # Data splitting utility
├── requirements.txt         # Project dependencies
├── setup_gpu_env.bat        # Automated setup script
└── README.md                # Project documentation
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
