import torch
from transformers import BertTokenizer


class QuestionClassifierDataset(torch.utils.data.Dataset):
    
    def __init__(self, df):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.encodings = [self.tokenizer(text, return_tensors='pt', max_length=50, padding=True, truncation=True) 
                          for text in df.iloc[:, 0]]
        self.labels = [0 if label == 'general' else 1 for label in df.iloc[:,1]]
        
    def __getitem__(self, idx):
        return self.encodings[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)