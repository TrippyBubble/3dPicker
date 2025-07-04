# 🦠 Colony Counter with YOLOv5

Этот проект предназначен для автоматического подсчёта **бактериальных колоний на чашках Петри** с помощью компьютерного зрения и модели **YOLOv5**. Модель проходит обучение на размеченных изображениях и предсказывает координаты колоний на новых фото.

---

## 📂 Структура проекта

```bash
colony_counter/
├── dataset/
│   ├── images/
│   │   ├── train/         # тренировочные изображения
│   │   └── val/           # валидационные изображения
│   ├── labels/
│   │   ├── train/         # аннотации .txt в формате YOLO (bbox)
│   │   └── val/
├── yolov5/                # YOLOv5-клон (автоматически клонируется)
├── dataset.yaml           # описание датасета
├── train.py               # обучение модели
├── detect_all.py          # запуск предсказаний на изображениях
├── graph.ipynb  # визуализация качества обучения
├── colony_predictions.csv # результат детекции (csv-таблица)
├── README.md              # ты читаешь его :)
```
## 🚀 Быстрый старт

### 1. Установи зависимости
```bash
pip install certifi opencv-python pandas matplotlib
```
Если yolov5/ нет — он автоматически клонируется при запуске train.py.

### 2. Подготовь данные
- Картинки: dataset/images/train/ и dataset/images/val/
- Аннотации: .txt в YOLO формате → dataset/labels/train/ и val/

**Формат аннотации (filename.txt):**
```angular2html
0 0.52 0.43 0.05 0.06
```

(где 0 — класс “colony”, а остальные — координаты bbox в YOLO-формате)

### 3. Обучение модели
```bash
python train.py
```
Модель сохранится по пути:
```angular2html
runs/train/colony_detection*/weights/best.pt
```
### 4. Предсказания на новых изображениях
```bash
python detect_all.py
```
Скрипт:
- автоматически найдёт best.pt
- выполнит распознавание всех изображений из dataset/images/val
- сохранит таблицу colony_predictions.csv с количеством колоний на каждом изображении

## ⚠️ Важные замечания
- yolov5/ — клонируется из https://github.com/ultralytics/yolov5 и не должен пушиться в git. Добавь его в .gitignore.
- Обязательно проверь качество разметки — это критически важно.
- Убедись, что имена файлов .jpg и .txt совпадают.
- Устанавливай model.conf = 0.5-0.6 для отсечения ложных срабатываний.

## 🧠 Необходимые улучшения и дальнейшее развитие
На данный момент в проекте наблюдается критический недостаток данных для обучения модели, а так же низкое качество разметки. Далее планируется развитие проекта: доабвление и улучшение данных для обучения модели.