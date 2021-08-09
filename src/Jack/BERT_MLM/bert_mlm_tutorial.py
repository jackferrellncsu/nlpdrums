from re import A
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.nn import Softmax
import statistics
import json
import numpy



brown_data = pd.read_csv("Data/brown_master.csv", sep = "\t")


train_cal, test = train_test_split(brown_data, test_size=0.25, shuffle=True)
train, cal = train_test_split(train_cal, test_size=0.03, shuffle=True)
train_cal = None
train = None

cal_clean_sents = cal["clean_sentences"].tolist()
cal_mask_inds = cal["mask_inds"].tolist()

# cal_cs_red = cal_clean_sents[1:10]
# cal_mi_red = cal_mask_inds[1:10]

from transformers import BertTokenizer, BertForMaskedLM
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


#input.input_ids is identical to performing following operations on every sentence:
#---------------------Following Tutorial--------------------------------------------------#
#Splits text into tokens
text = "[CLS] "+  cal_clean_sents[3] + " [SEP]"
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)
tokenized_text[5] = "[MASK]"
print(tokenized_text)
masked_index = tokenized_text.index("[MASK]")

#gets encoding/index for each token
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

#Create the segments tensors
segments_ids = [0] * len(tokenized_text)

#Convert inputs to pytorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

#Load pre-trained model (weights)
model =  BertForMaskedLM.from_pretrained("bert-base-uncased")
model.eval()

labels = torch.tensor([1]).unsqueeze(0)

with torch.no_grad():
    predictions = model(tokens_tensor, segments_tensors)


predicted_index = torch.argmax(predictions.logits[0][masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)
print(tokenized_text)
print(predicted_token)