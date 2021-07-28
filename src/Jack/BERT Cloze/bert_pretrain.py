from os import sched_get_priority_max
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers.utils.dummy_pt_objects import CanineLayer





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
text = "[CLS] "+  cal_cs_red[3] + " [SEP]"
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)
tokenized_text[20] = "[MASK]"
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
input = tokenizer(cal_clean_sents, return_tensors = "pt", max_length=512, truncation=True, padding="max_length")

#clone input_ids to get labels tensor
input["labels"] = input.input_ids.detach().clone()

mask_data(input, cal_mask_inds)

outputs = model(**input)


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