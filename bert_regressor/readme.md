# bert_regressor.BertRegressor


```python
class bert_regressor.BertRegressor(max_len=300, batch_size=20, epoch=1, lr=3e-5, optimizer=optim.Adam, from_pretrained=True)
```
## Note
Download "bert-base-chinese-model", "bert-base-chinese-tokenizer" and "pretrained_bert_params.pkl" from S3 and put them in "bert_regressor/".


## Parameters
#### **max_len: *int, default=300***
>  The maximum length (in number of tokens) for the inputs to the bert model.
#### **batch_size: *int, default=20***
> The number of samples in a batch.
#### **epoch: *int, default=1***
> The maximum number of passes over the training data.
#### **lr: *int, default=3e-5***
> Learning rate.
#### **optimizer: *str, default=optim.Adam***
> Defaut 'optim.Adam' will use `torch.optim.Adam`. Most optimization algorithms from Pytorch are supported.
#### **from_pretrained: *bool, default=True***
> True when use pretrained bert regression model otherwise False.
<br>

## *Attributes*
#### **device**
#### **pretrained_tokenizer_path**
#### **tokenizer**
#### **criterion**
<br>

## *Methods*
#### **fit** *(data, feature_col_name, target_col_name)*
> **data:** Training data. Only support pandas dataframe. <br>
> **feature_col_name:** Feature's column name of data.<br>
> **target_col_name:** Target's column name of data.<br>

#### **predict** *(data, feature_col_name)*
> **data:** Testing data or unlabeled data. Only support pandas dataframe. <br>
> **feature_col_name:** Feature's column name of data.<br>

#### **save** *(path, save_entire_model=False)*
> **path:** The location where you save the model.<br>
> **save_entire_model:** True when only save the model's parameters (recommend), otherwise False when save the entire model using pickle.
<br>
