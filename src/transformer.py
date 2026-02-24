import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
from torch.utils.data import DataLoader
import evaluate

from tqdm.auto import tqdm
from src.next_token_dataset import AuxDataset
from transformers import logging


def run_transformer_run(csv_file):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.set_verbosity_error()  # Отключаем предупреждения и отчёты, потому что больно

    # Загружаем модель и токенизатор
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(DEVICE)
    # tokenizer.pad_token = tokenizer.eos_token # Устанавливаем EOS-токен как pad-token

    # Читаем CSV файл
    df = pd.read_csv(csv_file) #!!!! 
    text_column = df['text']

    # Создаём набор данных и загрузчик
    dataset = AuxDataset(text_column, tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Создаем генератор
    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, device=DEVICE)

    hypotheses = []
    references = []
    # Основной цикл генерации
    for batch in tqdm(loader, desc="Processing Batches"):
        prompts = batch['prompt']
        refs = batch['reference']

        # Генерация
        outputs = generator(prompts,
                            max_new_tokens = 20, 
                            num_return_sequences=1,  # Количество последовательностей
                            do_sample=True,  
                            top_p=0.95,  
                            temperature=0.8
                        )

        # Продолжения
        continuations = [output[0]['generated_text'] for output in outputs]
        new_parts = []
        for prompt, continuation in zip(prompts, continuations):
            # Убираем входную подсказку из полной генерации
            new_part = continuation[len(prompt):].strip()  # len(prompt) даёт длину начальной фразы
            new_parts.append(new_part)
        hypotheses.extend(new_parts)
        references.extend(refs)

    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=hypotheses, references=references)
    print(f'\n\n{"-"*50}\n:жпт2 метрики руж:')
    print(results)


def gpt2_generate_sequence(csv_file):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем модель и токенизатор
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(DEVICE)

    # Читаем CSV файл ТОЛЬКО ДЛЯ ПЕРВЫХ nrows СТРОК
    df = pd.read_csv(csv_file, nrows=5)  # Только первые две(5) строки!!!
    text_column = df['text']

    # Создаём набор данных и загрузчик
    dataset = AuxDataset(text_column, tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)  # Бач-сайз ставим равным 1, чтобы соблюдать порядок строк

    # Создаем генератор
    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, device=DEVICE)

    # Основной цикл генерации (ограничен двумя примерами)
    for idx, batch in enumerate(loader):
        prompts = batch['prompt']
        reference = batch['reference'][0]  # Извлекаем референсную строку

        # Генерация
        outputs = generator(prompts,
                            max_new_tokens=20,
                            num_return_sequences=1,  # Количество последовательностей
                            do_sample=True,
                            top_p=0.95,
                            temperature=0.8
                           )

        # Получаем полное продолжение
        full_generation = outputs[0][0]['generated_text'].strip()

        # Выводим оригинальную последовательность и полную генерацию
        print(f'\n\n{"-"*50}\n:жпт2 пример #{idx+1}:')
        print(f'Исходный текст: {df['text'].iloc[idx]}')
        print(f'Входная последовательность: {prompts[0]}')
        print(f'Полный текст с продолжением: {full_generation}')
        print('-'*50)