import os
import re
import unicodedata
import emoji

import tqdm
from tqdm.auto import tqdm  # Используем auto для поддержки jupyter notebooks и консольных приложений
import json
import pandas as pd
from collections import Counter

from nltk.tokenize import word_tokenize

source_csv = 'data/raw_dataset.csv'
tweets_csv = 'data/tweets.csv'
preprocess_csv = 'data/dataset_processed.csv'

csv_files_to_check = [source_csv, tweets_csv, preprocess_csv]



def extract_texts_from_to(source_csv, destination):
    """Функция изуилечения текстов"""

    # А почему этим пидорам можно без заголовка файлы создавать? И все терпят и говорят какие они молодцы
    columns_names = ['wtf_0', 'wtf_1_id', 'wtf_2_date', 'wtf_3_noquerry', 'wtf_4_author', 'text']
    df = pd.read_csv(source_csv, header=None, names=columns_names, encoding='ISO-8859-1')

    num_rows = df.shape[0]
    num_columns = df.shape[1]
    print(f"Количество строк: {num_rows}")
    print(f"Количество столбцов: {num_columns}")

    sixth_column = df.iloc[:, 5] 
    sixth_column.to_csv(destination, index=False, header=True) # только  кукареканья петушков
    print("Кукареки сохранены в", destination)

    
# Чистим пёрышки
def clean_text(text):
    """Выкинуть хлам из дома и старых позвать друзей"""
    text = text.lower()                                                                     # Приведение текста к нижнему регистру
    text = re.sub(r'http\S+', '', text)                                                     # Удаление ссылок
    text = re.sub(r'@\S+', '', text)                                                        # Удаление упоминаний пользователей (@username)
    text = ''.join(c for c in text if c.isascii() or not emoji.is_emoji(c))                 # Удаление эмодзи, можно и оставить было бы, ради экспрмнтА
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')    # Нормализуем текст, избавлЯЯсь от непечатаемых символов 
    text = re.sub(r'\W+', ' ', text).strip()                                                # Простецкая очистка от лишних символов и двойных пробелов
    
    return text


def preprocess_data(input_filename, output_filename):
    """Функция предварительной обработки данных."""

    total_lines = sum(1 for _ in open(input_filename, 'r', encoding='utf-8'))
    total_lines_without_header = total_lines - 1

    with open(input_filename, 'r', encoding='utf-8') as infile, \
        open(output_filename, 'w', encoding='utf-8') as outfile:
        
        outfile.write("cleaned_texts\n")  # нормально пишем, один столбец - один заголовок
        
        headers = infile.readline() # да, есть заголовок, да, у единственного столбца и читаем в неиспользуемую переменную, йес 
        for line in tqdm(infile, total=total_lines_without_header, desc="Processing"):
            cleaned_line = clean_text(line.strip())  
            outfile.write(cleaned_line + "\n")       

    print("Данные были обработаны и сохранены в", output_filename)


def check_files_existence(files):
    """
    Проверяет существование всех указанных файлов.
    """
    for file in files:
        if not os.path.exists(file):
            return False
    return True





# ###############################################################################################################
# ###############################################################################################################

# ТАК БЛЭТ ДАЛЬШЕ НЕ ЧИТАЕМ ЭТО ДЛЯ РУКОПАШНОГО БОЯ

# ###############################################################################################################
# ###############################################################################################################


# Раскомментировать если токенизировать энэлтикеем 
# import nltk
# nltk.download('punkt')  # Скачиваем ресурс для токенизации
# nltk.download('punkt_tab')  # Скачиваем ресурс для токенизации

def nltk_tokenize_text(text):
    """Токенизирует текст на отдельные слова и знаки препинания NLTK"""
    tokens = word_tokenize(text)
    return ' '.join(tokens)  # Возвращаем токены в виде строки, разделённой пробелами, не ну а вдруг отличаются


def whitespace_tokenize_text(text):
    """Токенизийует текст по пьёбелам"""
    tokens = text.split()  # Делим строку по пробелам и тут же лепим всё обратно, апож
    return ' '.join(tokens)


    
def tqdm_tokenize_csv(cleaned_csv, whitepower_toks_csv):
    """Функция токенизации с отображением прогресса"""
    
    df = pd.read_csv(cleaned_csv)
    
    tqdm.pandas(desc="Tokenizing (Whitespace)")
    df['whitespace_tokenized_text'] = df['cleaned_texts'].progress_apply(whitespace_tokenize_text)
    df[['whitespace_tokenized_text']].to_csv(whitepower_toks_csv, index=False)
    
    df.to_csv(cleaned_csv, index=False) # датафрейма перезаписывает исходный файл, ай-ай, ой-ой
    print("Данные токенизированы и сохранены в", whitepower_toks_csv)



def create_labels(row):
    """Формирует целевую последовательность, сдвинутую на 1 токен вправо"""
    tokens = row['whitespace_tokenized_text'].split()
    shifted_tokens = tokens[1:]
    return ' '.join(shifted_tokens)

def add_targets(csv_filename):
    """Добавляет столбец target в CSV-файл"""
    df = pd.read_csv(csv_filename)
    
    tqdm.pandas(desc="Creating targets...")
    df['labels'] = df.progress_apply(create_labels, axis=1)
    df.to_csv(csv_filename, index=False) # Сохраняем изменённый датафрейм на зад в CSV
    print("Добавлены метки в", csv_filename)



def make_lists_of_texts_from(csv_file, save=False):
    """"""
    df = pd.read_csv(csv_file)
    # Выделяем токенизированные тексты и метки
    tokenized_texts = df['whitespace_tokenized_text'].tolist()
    labels = df['labels'].tolist()

    if save:
        with open('data/texts_lists.json', 'w') as f:
            data_to_save = {'texts': tokenized_texts, 'labels': labels}
            json.dump(data_to_save, f)
            print("Списки текстов сохранены в ","data/texts_lists.json " )
    print("Созданы списки текстов")
    return tokenized_texts, labels


def make_w2idx_dict_from(list_of_texts):
    """Создаём словарь из всех слов в CSV-файле"""

    # Строим словарь (mapping) токенов в индексы
    words_counter = Counter(word for text in list_of_texts for word in text.split())
    sorted_words = sorted(words_counter.keys())
    word2idx = {'<UNK>': 0, '<PAD>': 1}
    word2idx.update({w: i + 2 for i, w in enumerate(sorted_words)})
    print("Созданы word2idx словарик")
    return word2idx
   

def make_encoded_lists(tokenized_texts, labels, word2idx, save=False):
    """"""
    # Преобразовываем токены в индексы с прогрессбаром
    encoded_texts = [encode_tokens(str(text), word2idx) for text in tqdm(tokenized_texts, desc="Encoding texts")]
    encoded_labels = [encode_tokens(str(label), word2idx) for label in tqdm(labels, desc="Encoding labels")]
    
    if save:
        with open('data/encoded_texts_lists.json', 'w') as f:
            data_to_save = {'encoded_texts': encoded_texts,
                            'encoded_labels': encoded_labels}
            json.dump(data_to_save, f)
            print("Списки текстов с закодированными токенами сохранены в ","data/encoded_texts_lists.json " )
    print("Созданы списки  текстов с закодированными токенами")
    return encoded_texts, encoded_labels   


# Преобразуем токены в индексы
def encode_tokens(tokens_str, word2idx):
    """Строка токенов превращается в список индексов"""
    return [word2idx.get(w, 0) for w in tokens_str.split()]
