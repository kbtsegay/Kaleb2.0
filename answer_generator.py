import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer
from datasets import AnswerGeneratorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import sys

# This model answers "general knowledge" questions using finetuned GPT2

class GeneralQuestionResponder(nn.Module):

    def __init__(self):
        super(GeneralQuestionResponder, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.linear = nn.Linear(768, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids, attention_mask)
        return self.linear(outputs[0])
    
# TO DO
def train(model, train_data, val_data, batch_size=16, lr=1e-6, num_epochs=5):
    pass
