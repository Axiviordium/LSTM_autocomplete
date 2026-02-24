
import sys
sys.path.append('src')  # добавляем путь к директории src, чтобы Python смог-таки находить модули там

import tqdm
from tqdm.auto import tqdm  # Используем auto для поддержки jupyter notebooks и консольных приложений
import pandas as pd

from transformers import BertTokenizerFast
from transformers import set_seed
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from collections import defaultdict
import evaluate as ev

import src.next_token_dataset as ntd
import src.lstm_model as modl # [PAD] [PAD] [PAD] 

import json
import pickle

# Установка фиксированного seed для воспроизводимости результатов
set_seed(42)

# Набор гиперпараметров
BATCH_SIZE = 32
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
MAX_LENGTH = 55  # Максимальная длина токенизированного текста
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./models"    

train_texts_csv = 'data/train_samples.csv'
val_texts_csv = 'data/val_samples.csv'
test_texts_csv = 'data/test_samples.csv'

# Основная функция которая вообще
def run_lstm_experiment(csv_file):
    """Замесить и нарубить"""

    # *******************************************************
    # Читаем данные
    # *******************************************************
    df = pd.read_csv(csv_file, encoding="utf-8") # !!!
    texts = df['whitespace_tokenized_text'].tolist()
    print("Данные прочитаны")

    # *******************************************************
    # Разбиваем на выборки
    # *******************************************************
    train_texts, texts_temp = train_test_split(texts, test_size=0.2, random_state=42)  # Шаг 1: Разделяем на Train и Temp (80%/20%)
    val_texts,  test_texts = train_test_split(texts_temp, test_size=0.5, random_state=42) # Шаг 2: Разделяем Temp на Val и Test (50%/50%)
    print("Выполнено разбиение на выборки")

    save_to_csv(train_texts, train_texts_csv)
    save_to_csv(val_texts, val_texts_csv)
    save_to_csv(test_texts, test_texts_csv)
    print("Выборки сохранены")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    print("Токенайзер загружен")
    
    # Подготовка загрузчиков данных
    print("Создание наборов данных...")
    train_dataset = ntd.TextDataset(train_texts, tokenizer, MAX_LENGTH)
    val_dataset = ntd.TextDataset(val_texts, tokenizer, MAX_LENGTH)
    test_dataset = ntd.TextDataset(test_texts, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    print("Наборы данных и загрузчики созданы")

    # *******************************************************
    # Создаём и обучаем LSTM-модель
    # *******************************************************
    model = modl.TokenPredictionLSTM(tokenizer, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("Модель создана")

    print("Начинаем процесс обучения:")
    history = train_the_model(train_loader, val_loader, model, optimizer, tokenizer, epochs=EPOCHS)
    print("Обучение завершено")
    print('*' * 25)
    print()
    # Вывод метрик и 2 примеров:
    display_last_records(history)
    print('*' * 25)
    print()
    df = pd.read_csv(test_texts_csv)
    first_rows = df.head(5)["text"] 
    print('Тексты тестового набора:') 
    print(first_rows)
    for index, text in first_rows.items():
        start_seq, gen_seq = model.generate_sequence(text)
        print(f'\n\n{"-"*50}\nLSTM пример #{index+1}:')
        print(f"Исходный текст: {text}")
        print(f"Входная последовательность: {start_seq}\n Сгенерированная последовательность: {gen_seq}")
        print('-'*50)


def train_the_model(train_dataloader, val_dataloader, model, optimizer, tokenizer, epochs=EPOCHS):
    """"Цикл обучения"""
    global_step = 0
    criterion = nn.CrossEntropyLoss()
    history = defaultdict(list)

    for epoch in range(epochs):
        # 1. ТрениВовки
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        running_loss = 0.0
        for step, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            inputs = batch["input_ids"].to(DEVICE)
            outputs = model(inputs)
            logits = outputs.reshape(-1, outputs.size(-1))
            
            # сдвиг вправо для получения меток 
            shifted_targets = inputs[:, 1:]         # Убираем первый токен
            padded_targets = torch.cat([shifted_targets, torch.full((inputs.size(0), 1), fill_value=-100, dtype=torch.long, device=DEVICE)], dim=1) # добавляем специальную маску -100 для последнего элемента 
            flattened_targets = padded_targets.reshape(-1)
            
            loss = criterion(logits, flattened_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            global_step += 1 # забыл зачем ввёл, кто придумает - пусть уж допишет

        # Выводим среднюю потерю на данном этапе
        average_train_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}: Average Training Loss={average_train_loss:.4f}")
        history["train_loss"].append(average_train_loss)
        
        # 2. Оценка на валидационном наборе
        model.eval()
        val_loss = compute_average_loss(model, val_dataloader, criterion)
        print(f"Epoch {epoch+1}: Validation Loss={val_loss:.4f}")
        history["val_loss"].append(val_loss)
        results = compute_average_rouge(model, val_dataloader, tokenizer)
        history["rouge"].append(results)

        # Фотография, 9 х 12
        save_path = f'{SAVE_DIR}/model_epoch_{epoch + 1}.pth'
        model.save_checkpoint(epoch + 1, val_loss, optimizer, save_path)
        
        # Если прервётся ход истории, то историю пройденных эпох сохраним! 
        filename = f'{SAVE_DIR}/history.json'  # Или './history.pkl'
        save_training_history(history, filename)
    return history


def compute_average_loss(model, dataloader, criterion):
    """Лучшие потери даром"""
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to(DEVICE)
            outputs = model(inputs)
            logits = outputs.reshape(-1, outputs.size(-1))
            shifted_targets = inputs[:, 1:]         # Убираем первый токен
            padded_targets = torch.cat([shifted_targets, torch.full((inputs.size(0), 1), fill_value=-100, dtype=torch.long, device=DEVICE)], dim=1) # добавляем специальную маску -100 для последнего элемента 
            flattened_targets = padded_targets.reshape(-1)
            loss = criterion(logits, flattened_targets)
            losses.append(loss.item())
    return sum(losses) / len(dataloader)


def compute_average_rouge(model, dataloader, tokenizer):
    """
    Мулен руж
    """
    # Возможно оценка неправильная, если останутся силы перепроверю, учитываются ли PADлы или нет
    # Генерируем образцы и оцениваем ROUGE-метрику
    print("Расчёт метрик ROUGE...")
    samples = []
    refs = []
    for batch in dataloader:
        max_batch_len = batch["input_ids"].shape[1]             # Определяем  длину токенов в текущем батче
        split_point = int(max_batch_len * 0.75)                 # Высчитываем точку среза (3/4 длины)
        inputs = batch["input_ids"][:, :split_point].to(DEVICE) # Формируем ввод для модели, оставляя первые 1/4 последовательности
        predictions = model.batch_generate_tokens(inputs, max_batch_len - split_point)      # Генерируем продолжение
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True) # Декодируем полученные предсказания
        samples.extend(decoded_predictions)
        original_sequences = tokenizer.batch_decode(batch["input_ids"][:, split_point:], skip_special_tokens=True) # Формируем референсные последовательности, соответствующие последним 1/4 оригинала
        refs.extend(original_sequences)
    # Подсчет метрик ROUGE
    rouge = ev.load("rouge")
    results = rouge.compute(predictions=samples, references=refs)
    print("Расчёт метрик ROUGE завершён")
    return results


def save_to_csv(data, file_path):
    """
    Сохраняет переданные данные в CSV файл.
    """
    # Создаем DataFrame с одним столбцом
    df = pd.DataFrame({'text': data})
    
    df.to_csv(file_path, index=False, encoding='utf-8')
    print(f'Выборка успешно сохранена в файл {file_path}')


def save_training_history(history, filename):
    """
    Сохраняет историю обучения в файл.
    """
    ext = filename.split('.')[-1]
    if ext.lower() == 'json':
        with open(filename, 'w') as file:
            json.dump(history, file, indent=4)
    elif ext.lower() == 'pkl':
        with open(filename, 'wb') as file:
            pickle.dump(history, file)
    else:
        raise ValueError("Файл должен иметь расширение '.json' или '.pkl'.")


def load_training_history(filename):
    """
    Читает историю обучения из файла.
    """
    ext = filename.split('.')[-1]
    if ext.lower() == 'json':
        with open(filename, 'r') as file:
            return json.load(file)
    elif ext.lower() == 'pkl':
        with open(filename, 'rb') as file:
            return pickle.load(file)
    else:
        raise ValueError("Файл должен иметь расширение '.json' или '.pkl'.")


def display_last_records(history):
    """
    Отображает последние записи из словаря истории обучения.
    """
    print("\n--- Последние записи истории обучения ---")
    for key, values in history.items():
        last_record = values[-1] if values else None
        print(f"{key.capitalize()}:\t{last_record}")