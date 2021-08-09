from os import getpid
import pandas as pd
import random

#--------------------------Data Prep Helpers--------------------------------------------------#
#load in sentences df
#Not put back together yet, in (pre-bert) token form
def load_sentencesdf():
    raw_data = pd.read_csv("Data/brown.csv")
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
def whole_sentences(sentences):
    clean_sentences = sentences.apply(lambda x: x.str.cat(sep=" "), axis = 1)
    return clean_sentences

#gets index we will mask for each sentence
#one word is masked in each sentence 
def get_masked_inds(sentences):
    return sentences.apply(lambda x: get_mask_helper(x), axis = 1)
    
#-------------------------Actual Model Helpers------------------------------------------------#