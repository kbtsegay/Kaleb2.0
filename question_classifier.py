import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from datasets import QuestionClassifierDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import sys


# This classifier categorizes questions into two categories
# 1) Interview-style questions to get to know someone (2nd person)
# 2) General knowledge questions about the world and various facts (3rd person)

class QuestionClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(QuestionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask)
        embedding = bert_output['pooler_output']
        dropout_output = self.dropout(embedding)
        linear_output = self.linear(dropout_output)
        return self.relu(linear_output)


def train(model, train_data, val_data, batch_size=16, lr=1e-6, num_epochs=5):
    trainset, valset = QuestionClassifierDataset(train_data), QuestionClassifierDataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(valset, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    for epoch_num in range(num_epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:

                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f}')



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Error: Incorrect number of arguments. Expecting one.')
    elif sys.argv[1] == '--train':
        df = pd.read_excel('data/question_classification.xlsx')
        np.random.seed(24)
        df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df)), int(.9*len(df))])
        model = QuestionClassifier()
        train(model, df_train, df_val)
        pickle.dump(model, open('Kaleb2.0', 'wb'))
    elif sys.argv[1] == '--eval':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = pickle.load(open('Kaleb2.0', 'rb'))
        cont = True
        while(cont):
            question = input("Ask a question you'd like to classify: ")
            question = tokenizer(question, return_tensors='pt', max_length=50, padding=True, truncation=True)
            mask = question['attention_mask']
            id = question['input_ids']
            output = model(id, mask)[0]
            if output[0] > output[1]:
                print('Output: general question')
            else:
                print('Output: interview question')
            cont = input("Would you like to continue evaluating the model? (y/n): ")
            while cont.lower() not in ['y', 'n', 'yes', 'no']:
                cont = input("Invalid input. Would you like to continue evaluating the model? (y/n): ")
            if cont.lower() in ['y', 'yes']:
                cont = True
            else:
                cont = False
    else:
        print('Error: Invalid argument. Expecting \'--train\' or \'--eval\'.')
        