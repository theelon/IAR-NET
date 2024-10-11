# IAR-Net

## Overview
IAR-Net is a Human–Object Context Guided Action Recognition Network designed for monitoring industrial environments. This repository contains the implementation of the IAR-Net model, along with the LAMIS dataset,  and evaluation metrics used in the paper.

## Paper
- **Title**: IAR-Net: A Human–Object Context Guided Action Recognition Network for Industrial Environment Monitoring
- **Authors**: [Naval Kishore Mehta] , [Shyam Sunder Prasad] , [Sumeet Saurav] , [Ravi Saini] , and [Sanjay Singh]

## Features
- Human-object interaction modeling for improved action recognition.
- Context-aware action recognition in complex industrial environments.
- Designed for real-time monitoring and safety assurance.

## Requirements
- Python 3.4
- PyTorch >= 1.5.0
- CUDA >= 10.2
- Other dependencies in `requirements.txt`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/theelon/IAR-Net.git
   cd IAR-Net
2. pip install -r requirements.txt

## Evaluation
python evaluate.py --model checkpoint.pth --dataset test

If you use this code or the LAMIS dataset, please cite our paper:
@article{iar-net2024,
  title={IAR-Net: A Human–Object Context Guided Action Recognition Network for Industrial Environment Monitoring},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2024}
}
