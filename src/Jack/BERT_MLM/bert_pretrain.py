import enum
from os import sched_get_priority_max
from re import A
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers.utils.dummy_pt_objects import CanineLayer
import torch
from torch.utils.data import DataLoader
from torch.nn import Softmax
import math
import statistics
from tqdm import tqdm
import json
import numpy

def mask_data(token_tensor, mask_inds):
    #replace word @masked id with 103, index for [MASK] token 
    for i in range(token_tensor.input_ids.shape[0]):
        token_tensor.input_ids[i, mask_inds[i]] = 103

#Saves a list to a file in json format
def save_alphas(alphas, filename):
    path = "src/Jack/Data/" + filename
    with open(path, "w") as file:
        json.dump(alphas, file)

#loads a list from .txt file in json format
def load_cal_alphas(filename):
    path = "src/Jack/Data/" + filename
    with open(path, "r") as file:
        loaded_vals = json.load(file)
    return loaded_vals



brown_data = pd.read_csv("src/Jack/Data/brown_master.csv", sep = "\t")


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
#----------------------Real Work---------------------------------------------------------#
#Will work with input for big data moves:
input = tokenizer(cal_clean_sents[1], return_tensors = "pt", max_length=128, truncation=True, padding="max_length")
input["labels"] = input.input_ids.detach().clone()
input.labels
model =  BertForMaskedLM.from_pretrained("bert-base-uncased")
model.eval()

#clone input_ids to get labels tensor

from scipy import stats

mask_data(input, [cal_mask_inds[1]])

outputs = model(**input)

softmax = Softmax(dim=0)

res = softmax(outputs.logits[0][cal_mask_inds[1]]).tolist()
x  = res[input.labels[0][cal_mask_inds[1]]]
alphas_soft = numpy.array(alphas_soft)
p_val = len(numpy.nonzero(alphas_soft > x)[0].tolist()) / len(alphas_soft)
stats.percentileofscore(alphas_soft, x)


x = numpy.quantile(res, 0.45)
stats.percentileofscore(alphas_soft, x) 

len(res)
scat = numpy.array(res)
zed = numpy.nonzero(scat < 1)
zed[0].tolist()


p_val_true = len(numpy.nonzero(alphas_soft > true_soft)[0].tolist()) / len(alphas_soft)




pred = torch.argmax(res).item()

tokenizer.convert_ids_to_tokens(2120)
tokenizer.convert_ids_to_tokens(input.labels[0][cal_mask_inds[0]]).item()

#Takes max of all predictions minus probability (softmax output) of the true class
#max(y_all-y_t)
#FIXME: intervals are all the entire thing
def non_conf_infnorm(softmax_vector, true_index):
    true_prob = softmax_vector[true_index]
    arr = numpy.array(softmax_vector)
    arr = arr - true_prob
    arr = numpy.absolute(arr)
    return max(arr)

#Takes difference between max predicted untrue probability and true probability
def non_conf_maxdiff(softmax_vector, true_index):
    true_soft = softmax_vector[true_index]
    del softmax_vector[true_index]
    alpha = max(softmax_vector) - true_soft
    softmax_vector.insert(true_index, true_soft)
    return alpha




#Below for-loop calculates the nonconformity scores one at a time bc my stack overflows otherwise
#very slow, be warned
#Non conformity score is 1-softmax of result vector at true word, IE how likely BERT thought the true word was to occur
alphas_max = []
alphas_inf = []
alphas_soft = []

for i in range(len(cal_clean_sents)):
    input = tokenizer(cal_clean_sents[i], return_tensors = "pt", max_length=256, truncation=True, padding="max_length")
    input["labels"] = input.input_ids.detach().clone()
    mask_data(input, [cal_mask_inds[i]])

    outputs = model(**input)

    result_softmax = softmax(outputs.logits[0][cal_mask_inds[i]]).tolist()

    #Gets predicted prob of true word
    soft_true = result_softmax[input.labels[0][cal_mask_inds[i]]]

    #Gets index of true word
    true_index = input.labels[0][cal_mask_inds[i]]

    alphas_max.append(non_conf_maxdiff(result_softmax, true_index))
    alphas_inf.append(non_conf_infnorm(result_softmax, true_index))
    alphas_soft.append(1-soft_true)
    print(i)

#-----------------------------Working with conformal predictions-------------------------------------------------------#
alphas_max[:5]
alphas_inf[:5]
alphas_soft[:5]

#Save calibrated alphas
save_alphas(alphas_max, "max_alphas1k.txt")
save_alphas(alphas_inf, "inf_alphas1k.txt")
save_alphas(alphas_soft, "soft_alphas1k.txt")

from numpy import quantile

q = quantile(alphas_max, 0.95)

test_clean_sents = test["clean_sentences"].tolist()
test_mask_inds = test["mask_inds"].tolist()

epsilon = .9
q1 = quantile(alphas_max, epsilon)
q2 = quantile(alphas_inf, epsilon)

conf_intervals_max = []
conf_intervals_infnorm = []
true_inds = []


max_int_accuracy = 0
infnorm_int_accuracy = 0

import progressbar

for i in range(len(test_clean_sents[1:1000])):
    input = tokenizer(test_clean_sents[i], return_tensors = "pt", max_length=128, truncation=True, padding="max_length")
    input["labels"] = input.input_ids.detach().clone()
    mask_data(input, [test_mask_inds[i]])

    #Feed input into model
    outputs = model(**input)

    #Get output at the masked index, apply softmax
    result_softmax = softmax(outputs.logits[0][test_mask_inds[i]]).tolist()

    #Get true label index:
    true_ind = input.labels[0][test_mask_inds[i]]

    true_inds.append(true_ind)

    interval_max = []
    interval_infnorm = []
    for j in range(len(result_softmax)):
        alpha_1 = non_conf_maxdiff(result_softmax, j)
        if alpha_1 <= q1:
            interval_max.append(j)
        
        alpha_2 = non_conf_infnorm(result_softmax, j)
        if alpha_2 <= q2:
            interval_infnorm.append(j)
        print("j:", j)
    
    if true_ind in interval_max:
        max_int_accuracy += 1
    
    if true_ind in interval_infnorm:
        infnorm_int_accuracy += 1
        
    conf_intervals_max.append(interval_max)
    conf_intervals_infnorm.append(interval_infnorm)
    print(i)
    


alphas_soft = numpy.array(alphas_soft)

q_soft_90 = quantile(alphas_soft, 0.9)
q_soft_95 = quantile(alphas_soft, 0.95)
q_soft_80 = quantile(alphas_soft, .8)
q_soft_75 = quantile(alphas_soft, .75)
conf_intervals_soft_95 = []
conf_intervals_soft_90 = []
conf_intervals_soft_80 = []
conf_intervals_soft_75 = []

accuracy_95 = 0
accuracy_90 = 0
accuracy_80 = 0
accuracy_75 = 0

point_accuracy = 0

#Diagnostic Measures:
#contains p-values for all true predictions
p_vals_true = 0
p_vals_false = 0
creds = 0

for i in range(len(test_clean_sents[0:1000])):
    input = tokenizer(test_clean_sents[i], return_tensors = "pt", max_length=128, truncation=True, padding="max_length")
    input["labels"] = input.input_ids.detach().clone()
    mask_data(input, [test_mask_inds[i]])

    #Feed input into model
    outputs = model(**input)

    #Get output at the masked index, apply softmax
    result_softmax = softmax(outputs.logits[0][test_mask_inds[i]]).tolist()
    result_softmax = numpy.array(result_softmax)

    #Predicted word- word with highest softmax score
    
    
    #Do 1-all softmaxes to get nonconf scores
    non_confs = 1 - result_softmax

    #Get nonconf score for true and predicted:
    true_ind = input.labels[0][test_mask_inds[i]]
    pred_ind = numpy.argmax(result_softmax)

    true_nonconf = non_confs[true_ind]
    pred_nonconf = non_confs[pred_ind]

    #Find where nonconf is less than the epsilon quantile:
    region_inds_95 = numpy.nonzero(non_confs <= q_soft_95)[0].tolist()
    region_inds_90 = numpy.nonzero(non_confs <= q_soft_90)[0].tolist()
    region_inds_80 = numpy.nonzero(non_confs <= q_soft_80)[0].tolist()
    region_inds_75 = numpy.nonzero(non_confs <= q_soft_75)[0].tolist()

    conf_intervals_soft_95.append(region_inds_95)
    conf_intervals_soft_90.append(region_inds_90)
    conf_intervals_soft_80.append(region_inds_80)
    conf_intervals_soft_75.append(region_inds_75)

    #Get true label index:
    
    true_soft = result_softmax[true_ind]

    #Add p-value of prediction for true val
    p_vals_true += len(numpy.nonzero(alphas_soft > true_nonconf)[0].tolist()) / len(alphas_soft)

    #Add p-values of false predictions
    p_vals_false = sum_all_pvals(numpy.delete(non_confs, true_ind), alphas_soft)

    #Add p-value of highest softmax:
    creds += len(numpy.nonzero(alphas_soft > pred_nonconf)[0].tolist()) / len(alphas_soft)

    if true_ind == pred_ind:
        point_accuracy += 1

    if true_ind in region_inds_95:
        accuracy_95 += 1
    
    if true_ind in region_inds_90:
        accuracy_90 += 1
    
    if true_ind in region_inds_80:
        accuracy_80 += 1
    
    if true_ind in region_inds_75:
        accuracy_75 += 1

    true_inds.append(true_ind)

    
    print(i)
    


lengths_95 = [len(i) for i in conf_intervals_soft_95]
lengths_90 = [len(i) for i in conf_intervals_soft_90]
lengths_80 = [len(i) for i in conf_intervals_soft_80]
lengths_75 = [len(i) for i in conf_intervals_soft_75]


statistics.mean(lengths_95)
statistics.mean(lengths_90)
statistics.mean(lengths_80)
statistics.mean(lengths_75)

statistics.median(lengths_95)
statistics.median(lengths_90)
statistics.median(lengths_80)
statistics.median(lengths_75)

emp_conf_95 = accuracy_95 / len(lengths_95)
emp_conf_90 = accuracy_90 / len(lengths_95)
emp_conf_80 = accuracy_80 / len(lengths_95)
emp_conf_75 = accuracy_75 / len(lengths_95)

print(emp_conf_95)
print(emp_conf_90)
print(emp_conf_80)
print(emp_conf_75)

classification_accuracy = point_accuracy / len(lengths_95)

OP = p_vals_true / len(lengths_95)
OF = p_vals_false / len(lengths_95)

credibility = creds / len(lengths_95)

import matplotlib.pyplot as plt
from matplotlib import colors
fig = plt
fig.xlabel("Non-conformity Scores")
fig.hist(alphas_soft, color = 'r')
fig.show()

x1 = [emp_conf_75, emp_conf_80, emp_conf_90, emp_conf_95]
y1 = [.75, .8, .9, .95]
x2 = [.75, .8, .9, .95]
y2 = [.75, .8, .9, .95]
fig2 = plt
fig2.plot(x1, y1, label = "Empirical")
fig2.plot(x2, y2, label = "Proposed")
fig2.xlim()
fig2.legend()
fig2.grid()
fig2.show()

fig3 = plt
fig3.hist(lengths_95, color='r')
fig3.show()


def conf_pred_soft(test_df, model, tokenizer, alphas, epsilon):
    
    q = quantile(alphas, epsilon)

    conf_interval = []
    interval_accuracy = 0

    sentences = test_df["clean_sentences"].tolist()
    mask_inds = test_df["mask_inds"].tolist()

    conf_intervals = []
    interval_accuracy = 0
    point_accuracy = 0

    for i in range(len(sentences)):
        input = tokenizer(test_clean_sents[i], return_tensors = "pt", max_length=128, truncation=True, padding="max_length")
        input["labels"] = input.input_ids.detach().clone()
        mask_data(input, [test_mask_inds[i]])

        #Feed input into model
        outputs = model(**input)

        #Get output at the masked index, apply softmax
        result_softmax = softmax(outputs.logits[0][test_mask_inds[i]]).tolist()
        result_softmax = numpy.array(result_softmax)
        pred_ind = numpy.argmax(result_softmax)
        non_confs = 1 - result_softmax

        region_inds = numpy.nonzero(non_confs <= q)[0].tolist() 
        conf_intervals.append(region_inds)

        true_ind = input.labels[0][test_mask_inds[i]]

        if true_ind in region_inds:
            interval_accuracy += 1

        if true_ind == pred_ind:
            point_accuracy += 1

        if i % 100 == 0:
            print(i)
    
    acc = interval_accuracy / len(conf_intervals)
    point_acc = point_accuracy / len(sentences)
    return conf_intervals, acc, point_acc

#Takes in numpy array of nonconformity scores (except for true word) and calibration noncomf scores
#Returns sum of their p-values
def sum_all_pvals(nonconfs, cal_alphas):
    sum = 0
    for i in range(len(nonconfs)):
        sum += len(numpy.nonzero(cal_alphas > nonconfs[i])[0].tolist()) / len(cal_alphas)
    
    return sum