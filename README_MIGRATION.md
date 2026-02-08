# How to Train on Another Machine (with GPU)

## 1. Prerequisites
On the new laptop (your friend's laptop with RTX 3050):
1.  **Install Python 3.10 or 3.11**: Download from [python.org](https://www.python.org/downloads/).
    *   **Important**: Check the box "Add Python to PATH" during installation.
2.  **Install Git**: (Optional, if you want to clone via git).

## 2. Copy the Project
Copy the entire `tomato-disease-detector` folder to the new laptop.

## 3. Set Up the Environment (Automated)
I have created a script called `setup_gpu_env.bat` in this folder.
1.  Open the folder in File Explorer.
2.  Double-click `setup_gpu_env.bat`.
3.  This script will:
    *   Create a virtual environment.
    *   Install PyTorch with CUDA support (for RTX 3050).
    *   Install all other dependencies.

## 4. Run Training
1.  Open a terminal (cmd or PowerShell) in the folder.
2.  Activate the environment:
    ```cmd
    venv\Scripts\activate
    ```
3.  Run the training script:
    ```cmd
    python train.py
    ```
4.  The training should be fast (~15-20 mins). Wait for it to finish.
5.  It will verify the accuracy on the test set at the end.

## 5. Test the App
Once training is done:
1.  Run the app:
    ```cmd
    python app/main.py
    ```
2.  Open browser at `http://localhost:8000`.
3.  Upload an image to test prediction.

## 6. Copy Model Back (Optional)
If you want to run the app on your original laptop later:
1.  Copy the `model/efficientnet_b3_tomato.pth` file from the new laptop back to your original laptop.
2.  Place it in the `model/` folder.
