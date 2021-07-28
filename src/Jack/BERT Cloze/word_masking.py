from itertools import count
from numpy import core, result_type
from torch._C import device
from transformers import BertTokenizer, BertForMaskedLM
import torch
import pandas as p
import random
from pandas.core.series import Series 
from transformers import TrainingArguments
from transformers import Trainer
from sklearn.model_selection import train_test_split as tt_split



random.seed(24)

master = p.read_csv("src/Jack/Data/brown_master.csv", sep = "\t")



#Bert pieces initializing 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

#Tokenize words
#For now, using bert base input size of 512
#warning: Pretty slow, only run when necessary
inputs = tokenizer(clean_sentences, return_tensors = "pt", max_length=512, truncation=True, padding="max_length")

#clone input_ids to get labels tensor
inputs["labels"] = inputs.input_ids.detach().clone()

#get indexes of masked word in each sentence
masked_inds = get_masked_inds(sentences)
len(masked_inds)

#replace words at masked_ind with the mask token, which has ID 103
for i in range(inputs.input_ids.shape[0]):
    inputs.input_ids[i, masked_inds[i]] = 103


class BrownDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings.input_ids)

dataset = BrownDataset(inputs)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)



args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=16,
    num_train_epochs=2
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset
)

trainer.train()
















