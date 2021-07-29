from os import sched_get_priority_max
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers.utils.dummy_pt_objects import CanineLayer
import torch
from torch.utils.data import DataLoader
import math





brown_data = pd.read_csv("src/Jack/Data/brown_master.csv", sep = "\t")


train_cal, test = train_test_split(brown_data, test_size=0.25, shuffle=True)
train, cal = train_test_split(train_cal, test_size=0.1, shuffle=True)

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
#----------------------Real Work---------------------------------------------------------#
#Will work with input for big data moves:
input = tokenizer(cal_clean_sents[1:50], return_tensors = "pt", max_length=128, truncation=True, padding="max_length")


#clone input_ids to get labels tensor
input["labels"] = input.input_ids.detach().clone()

mask_data(input, cal_mask_inds[1:50])

class BrownDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def _getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings.input_ids)

calibration_dataset = BrownDataset(input)

cal_dl = DataLoader(calibration_dataset, batch_size = 50)

outputs = model(**input)

math.ceil(len(cal_clean_sents) / 50)

sent_number = 0
model_outputs = {}

for i in range(math.ceil(len(cal_clean_sents) / 50)):
    begin = i*50
    end = (i+1)*50
    if i >= math.ceil(len(cal_clean_sents) / 50)-1:
        i = len(cal_clean_sents )-1
        end = len(cal_clean_sents)
    
    batch = cal_clean_sents[begin:end]
    batch_inds = cal_mask_inds[begin:end]

    input = tokenizer(batch, return_tensors = "pt", max_length=128, truncation=True, padding="max_length")
    input["labels"] = input.input_ids.detach().clone()

    mask_data(input, batch_inds)

    outputs  = model(**input)
    model_outputs[sent_number] = outputs.logits[sent_number][batch_inds[sent_number]]
    sent_number += 1


    


    




predicted_outputs = {}
for i in range(outputs.logits.shape[0]):
    predicted_index = torch.argmax(outputs.logits[i][cal_mi_red[i]]).item()
    predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)

    predicted_indeces = torch.topk(outputs.logits[i][cal_mi_red[i]], 5)[1].tolist()
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indeces)

    predicted_outputs[i] = predicted_token

unsoftmax_output = {}
for i in range(outputs.logits.shape[0]):
    unsoftmax_output[i] = outputs.logits[i][cal_mi_red[i]]

def mask_data(token_tensor, mask_inds):
    #replace word @masked id with 103, index for [MASK] token 
    for i in range(token_tensor.input_ids.shape[0]):
        token_tensor.input_ids[i, mask_inds[i]] = 103

i = torch.argmax(predictions.logits[0][1]).item()
t = tokenizer.convert_ids_to_tokens(i)
print(t)