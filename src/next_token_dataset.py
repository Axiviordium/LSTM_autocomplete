from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.tokenized_data = tokenizer(data, padding=True, truncation=True, max_length=max_len, return_tensors="pt")

    def __getitem__(self, idx):
        """Возвращает список токенов для одного текста"""
        item = {'input_ids': self.tokenized_data.input_ids[idx],
                'attention_mask': self.tokenized_data.attention_mask[idx] #пока просто, чтобы осознать, что токенизатор так умеет
        }
        return item
    
    def __len__(self):
        """Возвращает длину списка текстов"""
        return len(self.tokenized_data.input_ids)
    

def split_text(text, tokenizer):
    """Вершки и корешки"""
    tokenized_text = tokenizer.encode(text)

    # Определяем границу разделения
    split_point = int(len(tokenized_text) * 0.75)

    first_part_tokens = tokenized_text[:split_point]
    second_part_tokens = tokenized_text[split_point:]

    first_part = tokenizer.decode(first_part_tokens)
    second_part = tokenizer.decode(second_part_tokens)

    return first_part.strip(), second_part.strip()


class AuxDataset(Dataset):
    def __init__(self, text_column, tokenizer):
        self.tokenizer = tokenizer  # Сохраняем токенизатор как экземпляр класса
        # Внутренняя функция-разделитель
        def splitter(text):
            return split_text(text, self.tokenizer)
        
        self.texts = text_column.apply(splitter).tolist()

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        prompt, reference = self.texts[idx]
        return {"prompt": prompt, "reference": reference}

