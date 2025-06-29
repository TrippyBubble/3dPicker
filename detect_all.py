import ssl
import certifi
import torch
import cv2
import pandas as pd
from pathlib import Path
import os

# Настройка безопасного SSL-контекста
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# Автоматически ищем последний вес best.pt в runs/train/
weights_paths = sorted(Path('runs/train/').glob('colony_detection_v*/weights/best.pt'), key=os.path.getmtime)
if not weights_paths:
    raise FileNotFoundError("Не найден файл best.pt в runs/train/")
weights_path = weights_paths[-1]
print(f"Используем модель: {weights_path}")

# Загружаем модель
model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(weights_path), force_reload=True)
model.conf = 0.6

# Путь к папке с валидационными изображениями
val_folder = Path('dataset/images/val')

results_list = []

# Проходим по всем изображениям в val
for img_path in sorted(val_folder.glob("*.[jp][pn]g")):
    img = cv2.imread(str(img_path))[:, :, ::-1]  # BGR → RGB
    results = model(img)
    count = len(results.xyxy[0])
    results_list.append({'filename': img_path.name, 'colonies': count})
    print(f"{img_path.name}: найдено колоний {count}")

# Сохраняем результаты в CSV
df = pd.DataFrame(results_list)
csv_path = 'colony_predictions.csv'
df.to_csv(csv_path, index=False)
print(f"Результаты сохранены в {csv_path}")