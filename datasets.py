import torch
from transformers import BertTokenizer, GPT2Tokenizer

# this produces a dataset compatible with torch.utils.data.DataLoader 
# for use with our BERT classifier

class QuestionClassifierDataset(torch.utils.data.Dataset):
    
    def __init__(self, df):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.encodings = [self.tokenizer(text, return_tensors='pt', max_length=50, padding=True, truncation=True) 
                          for text in df.iloc[:, 0]]
        self.labels = [0 if label == 'general' else 1 for label in df.iloc[:,1]]
        
    def __getitem__(self, idx):
        return self.encodings[idx], self.labels[idx]

    def __len__(self):
        return len(self.encodings)


# this produces a torch.utils.data.DataLoader dataset for use with 
# our GPT2-based answer generator

class AnswerGeneratorDataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.input_encodings = [self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                                for text in df.iloc[:, 0]]
        self.output_encodings = [self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                                for text in df.iloc[:, 1]] 
        
    def __getitem__(self, idx):
        return self.input_encodings[idx], self.output_encodings[idx]
    
    def __len__(self):
        return len(self.input_encodings)
    
