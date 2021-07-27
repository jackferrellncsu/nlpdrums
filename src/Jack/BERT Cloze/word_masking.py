from itertools import count
from numpy import core, result_type
from transformers import BertTokenizer, BertForMaskedLM
import torch
import pandas as p
import random
from pandas.core.series import Series 

#load cleaned data from file:
def load_data():
    with open("src/Jack/Data/brown_string.txt", "r") as file:
        corpus = file.read()
    return corpus

corpus = load_data()

#Bert pieces initializing 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

inputs = tokenizer(corpus, return_tensors = "pt")
inputs.keys()

inputs["labels"] = inputs.input_ids.detach().clone()






    













