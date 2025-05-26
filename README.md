# PPE Detection 🦺🧠

This project detects the usage of Personal Protective Equipment (PPE) such as helmets, vests, and masks in real-time using a deep learning model. It is optimized to run on NVIDIA GPUs using CUDA for acceleration.

## Features ✅

- Real-time PPE detection ⚡  
- Detects Helmet, Vest, and Mask ⛑  
- Supports images, videos, and live webcam feeds 🎥  
- CUDA-enabled for high-speed inference on NVIDIA GPUs 🚀  

Screenshots 🖼

![Screenshot 2025-05-26 1433522](https://github.com/user-attachments/assets/2051a1ef-61cb-4c3d-88e9-029025d842d5)

## Requirements 📦

- Python 3.7+
- NVIDIA GPU with CUDA & cuDNN installed
- Python dependencies listed in requirements.txt

Install dependencies:
```bash
pip install -r requirements.txt
```

## CUDA Setup ⚙

Make sure CUDA is installed and available:
```bash
import torch
print(torch.cuda.is_available())  # Should return True
```
