import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ._dataset import *
from ._models import *


class BertRegressor:
    def __init__(self, max_len=300, batch_size=20, epoch=1, lr=3e-5,
                 optimizer=optim.Adam, from_pretrained=True):
        self.max_len = max_len
        self.batch_size = batch_size
        self.epoch = epoch
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.pretrained_tokenizer_path = os.path.join(
            os.path.abspath(
                os.path.dirname(__file__)), 'bert-base-chinese-tokenizer')
        self.tokenizer = BertTokenizer.from_pretrained(
            self.pretrained_tokenizer_path)

        self.model = BertLinearModel().to(self.device)
        if from_pretrained:
            self.model.load_state_dict(torch.load(
                os.path.join(
                    os.path.abspath(os.path.dirname(__file__)),
                    'pretrained_bert_params.pkl'),
                map_location=self.device))
            print(f'Use Pretrained Model')
        else:
            print('Use bert-base-chinese Model')

        self.criterion = nn.MSELoss()
        self.learning_rate = lr
        self.optimizer = optimizer(
            self.model.parameters(), lr=self.learning_rate)

    def fit(self, data, feature_col_name, target_col_name):
        train_dataset = NewsDataset(
            data=data,
            feature_col_name=feature_col_name,
            target_col_name=target_col_name,
            tokenizer=self.tokenizer,
            mode='train',
            max_len=self.max_len
        )
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        for e in range(self.epoch):
            print(f'\nEpoch {e+1}/{self.epoch}')
            train(self.model, self.criterion, self.optimizer,
                  train_data_loader, self.device)

    def predict(self, data, feature_col_name):
        valid_dataset = NewsDataset(
            data=data,
            tokenizer=self.tokenizer,
            feature_col_name=feature_col_name,
            mode='predict',
            max_len=self.max_len
        )
        valid_data_loader = DataLoader(
            valid_dataset,
            shuffle=False,
        )

        predictions = predict(
            self.model, self.criterion, valid_data_loader, self.device)

        return np.array(predictions)

    def save(self, path, save_entire_model=False):
        if save_entire_model:
            torch.save(self.model, path)  # save the whole model
        else:
            torch.save(self.model.state_dict(), path)
