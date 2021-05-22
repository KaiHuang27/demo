import torch

from transformers import BertTokenizer
from torch.utils.data import Dataset


class NewsDataset(Dataset):
    def __init__(self, data, mode, tokenizer, max_len,
                 feature_col_name, target_col_name=None):
        assert mode in ['train', 'predict']
        self.mode = mode
        self.df = data.reset_index(drop=True)
        self.feature_col_name = feature_col_name
        self.target_col_name = target_col_name
        self.len = len(self.df)
        self.tokenizer = BertTokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        text = self.df.loc[idx, self.feature_col_name]
        token = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return_items = {'input_ids': token['input_ids'][0],
                        'token_type_ids': token['token_type_ids'][0],
                        'attention_mask': token['token_type_ids'][0]}
        if self.mode == 'train':
            return_items['targets'] = torch.tensor(
                self.df.loc[idx, self.target_col_name], dtype=torch.float
            ).view(1)

        return return_items

    def __len__(self):
        return self.len