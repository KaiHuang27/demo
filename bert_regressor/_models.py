import os
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import BertModel
#from transformers import BertConfig


class BertLinearModel(nn.Module):
    def __init__(self):
        super(BertLinearModel, self).__init__()
        self.bert_layer = BertModel.from_pretrained(
            os.path.join(os.path.abspath(
                os.path.dirname(__file__)), 'bert-base-chinese-model')
        )
        self.linear = nn.Linear(768, 256)
        self.out = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, token_type_ids, attention_mask):
        _, bert_out = self.bert_layer(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        drop_out = self.dropout(bert_out)
        l_out = self.relu(self.linear(drop_out))
        output = self.out(l_out)
        return output


def train(model, criterion, optimizer, data_loader, device='cpu'):
    model.train()
    pbar = tqdm(data_loader)
    for d in pbar:
        input_ids = d['input_ids'].to(device, dtype=torch.long)
        token_type_ids = d['token_type_ids'].to(device, dtype=torch.long)
        attention_mask = d['attention_mask'].to(device, dtype=torch.long)
        targets = d['targets'].to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask)

        # Computing Loss
        loss = criterion(outputs, targets)

        # Backpropagation
        loss.backward()

        # Optimizing
        optimizer.step()

        # show loss on processing bar
        pbar.set_postfix({'loss': loss.item()})


def predict(model, criterion, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for d in tqdm(data_loader):
            input_ids = d['input_ids'].to(device, dtype=torch.long)
            token_type_ids = d['token_type_ids'].to(device, dtype=torch.long)
            attention_mask = d['attention_mask'].to(device, dtype=torch.long)

            outputs = model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
            predictions.append(outputs.cpu().numpy()[0][0])

        return predictions
