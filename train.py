import os
import subprocess

import certifi
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
import ssl
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = ssl._create_unverified_context
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context



# Теперь можно безопасно делать:
import torch


if not os.path.exists('yolov5'):
    subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git'], check=True)
    subprocess.run(['pip', 'install', '-r', 'yolov5/requirements.txt'], check=True)

print("Обучаем YOLOv5 улучшенной конфигурацией...")

subprocess.run([
    'python', 'yolov5/train.py',
    '--img', '768',
    '--batch', '8',
    '--epochs', '150',
    '--data', 'dataset.yaml',
    '--weights', 'yolov5m.pt',
    '--project', 'runs/train',
    '--name', 'colony_detection_v2',
    '--cache',
    '--patience', '20',
    '--hyp', 'yolov5/data/hyps/hyp.scratch-low.yaml'
])
