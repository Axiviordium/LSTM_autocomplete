import torch
import torch.nn as nn
import pandas as pd

device = torch.device("cpu")

class TokenPredictionLSTM(nn.Module):
    def __init__(self, tokenizer, embedding_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.vocab_size)
    

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output, _ = self.lstm(embedded)
        prediction = self.fc(output)
        return prediction
    

    def generate_sequence(self, text, max_length=25):
        """
        Генерирует последовательность, начиная с предпоследнего токена строки из CSV-файла.
        :param max_length: Максимальная длина генерируемой последовательности.
        """

        tokens = self.tokenizer.encode(text)
        print(f"\nTokens: {tokens}, {self.tokenizer.decode(tokens)}")
        # Разделяем токены на две части: первые 3/4 и последние 1/4
        boundary = int(len(tokens) * 0.75)
        first_group = tokens[:boundary]
        print(f'First group: {first_group}, {self.tokenizer.decode(first_group)}')
        
        # Последний токен первой группы станет начальным токеном для генерации
        initial_token_id = first_group[-1]
        print(f'Init tok: {initial_token_id}, {self.tokenizer.decode(initial_token_id)}')

        generated_sequence = []
        current_input = torch.tensor([[initial_token_id]], dtype=torch.long)
        
        for i in range(max_length):
            pred = self.forward(current_input).argmax(dim=-1)[:,-1].item()
            generated_sequence.append(pred)
            
            if pred == self.tokenizer.eos_token_id or len(generated_sequence) >= max_length:
                break
                
            # Формируем новый вход путем добавления последнего предиктивного токена
            new_input_tensor = torch.tensor([[pred]])
            current_input = torch.cat([current_input, new_input_tensor], dim=1)
        
        return self.tokenizer.decode(first_group), self.tokenizer.decode(generated_sequence)


    def generate_tokens(self, input_ids, max_length=50):
        """Метод генерации продолжения текста"""
        generated_tokens = []
        with torch.no_grad():
            while True:
                outputs = self.forward(input_ids.to(device))[:, -1:, :]
                next_token = outputs.argmax(dim=-1)
                generated_tokens.append(next_token.squeeze().item())
                input_ids = torch.cat((input_ids, next_token), dim=1)
                if (len(generated_tokens) > max_length 
                    or next_token.item() == self.tokenizer.eos_token_id):
                    break
        return generated_tokens
    

    def batch_generate_tokens(self, input_ids, max_length=50):
        batch_size = input_ids.size(0)
        generated_tokens = []
        current_input = input_ids.clone().detach()
        
        for _ in range(max_length):
            outputs = self.forward(current_input)
            next_token_probs = outputs[:, -1, :]  # Вероятности последнего токена
            next_tokens = next_token_probs.argmax(dim=-1)  # Наиболее вероятные токены
            generated_tokens.append(next_tokens)  # Переходим в NumPy
            
            # Присоединяем новые токены к текущему входу
            current_input = torch.cat([current_input, next_tokens.unsqueeze(1)], dim=1)
        
        # Объединяем токены по оси batches
        result = torch.stack(generated_tokens, dim=1)
        return result
    
    def save_model(self, path):
        """
        Сохранение состояния модели в файл.
        """
        torch.save(self.state_dict(), path)
    

    def load_model(self, path):
        """
        Загрузка состояния модели из файла.
        """
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)


    def save_checkpoint(self, epoch, best_loss, optimizer, CHECKPOINT_PATH):
        """
        Сохраняет чекпоинт обучения, включая модель, оптимизатор и ключевые метрики.
        
        Аргументы:
            epoch (int): текущая эпоха обучения.
            best_loss (float): наилучшая потеря (ммм, наилучшая потеря) на данном этапе.
            best_accuracy (float): наивысшая точность на данном этапе. - Потом добавлю, если силы останутся
            optimizer (torch.optim.Optimizer): текущий оптимизатор.
            CHECKPOINT_PATH (str): путь к файлу для сохранения чекпоинта.
        """
        print("Сохранение всякого 18+...")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),  # сохраним состояние самой модели
            'optimizer_state_dict': optimizer.state_dict(),  # сохраним состояние оптимизатора
            'loss': best_loss
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        print(f'Создана промежуточная точка восстановления модели {CHECKPOINT_PATH}')


    def load_checkpoint(self, CHECKPOINT_PATH, optimizer):
      """
      Восстанавление состояния модели и оптимизатора.
      
      Аргументы:
          CHECKPOINT_PATH (str): путь к файлу чекпоинта.
          optimizer (torch.optim.Optimizer): текущий оптимизатор, чье состояние нужно восстановить.
      """
      try:
          checkpoint = torch.load(CHECKPOINT_PATH)
          self.load_state_dict(checkpoint['model_state_dict'])    # восстановление состояния модели
          optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    # восстановление состояния оптимизатора
          start_epoch = checkpoint['epoch'] + 1    # следующая эпоха после последней сохраненной
          best_loss = checkpoint.get('loss', float('inf'))    # начальная потеря
        #   best_accuracy = checkpoint.get('accuracy', 0.)    # начальная точность
          print(f"Восстановление из файла: '{CHECKPOINT_PATH}' эпоха: {start_epoch}")
          return start_epoch, best_loss   #, best_accuracy
      except FileNotFoundError:
          print(f"Файл восстановления не найден '{CHECKPOINT_PATH}'. Начинаем с САМОГО начала.")
          return 0, None