#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Step 1: Install Python dependencies from requirements.txt if it exists
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found, skipping Python dependency installation."
fi

# Step 2: Clone the Zonos repo
echo "Cloning Zonos repository..."
git clone https://github.com/Zyphra/Zonos.git

# Step 3: Install Zonos in editable mode
echo "Installing Zonos in editable mode..."
cd Zonos
pip install -e .

# Step 4: Install system dependency
echo "Installing espeak-ng..."
apt update
apt install -y espeak-ng

# Step 5: Clone the Chatterbox repo
cd ..

git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox
pip install -e .

cd ..

apt update && apt install -y ffmpeg
pip install --force-reinstall --no-cache-dir torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo "Setup completed successfully!"
