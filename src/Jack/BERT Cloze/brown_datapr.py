from json import load
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split


#load in sentences df
#Not put back together yet, in (pre-bert) token form
def load_sentencesdf():
    raw_data = pd.read_csv("src/Jack/Data/brown.csv")
    sentences = raw_data["raw_text"].str.split(expand=True)
    sentences = sentences.apply(lambda x: x.str.rsplit("/").str[0])
    sentences.insert(180, "mask_ind", None)
    return sentences
    

#Get index we will later mask, only called in 
def get_mask_helper(row):
    #end and beginning offset by one due to [CLS] token when making bert embeddings
    range = row.notna().sum()+1
    ind = random.randrange(1, range)

    return ind


#puts sentences back together, returns a list of them
def whole_sentences(sentence_df):
    k = clean_sentences = sentences.apply(lambda x: x.str.cat(sep=" "), axis = 1)
    return clean_sentences
    
#gets index we will mask for each sentence
#one word is masked in each sentence 
def get_masked_inds(sentences):
    return sentences.apply(lambda x: get_mask_helper(x), axis = 1)
    
#Apply functions
sentences = load_sentencesdf()
clean_sentences = whole_sentences(sentences)
clean_sentences.head()
mask_inds = get_masked_inds(sentences)
mask_inds.head()

#Create new DF with cleaned sentences and masked inds
new_df = pd.DataFrame([clean_sentences, mask_inds]).transpose()
new_df.head()

train_cal, test = train_test_split(new_df, test_size=0.25, shuffle=True)
train, cal = train_test_split(train_cal, test_size=0.25, shuffle=True)

#Save new dfs with sentences and masked inds as tab seperated CSV
new_df.to_csv("src/Jack/Data/brown_master.csv", sep = "\t", index = False)
train.to_csv("src/Jack/Data/brown_train.csv", sep = "\t", index = False)
test.to_csv("src/Jack/Data/brown_test.csv", sep = "\t", index = False)
cal.to_csv("src/Jack/Data/brown_cal.csv", sep = "\t", index = False)