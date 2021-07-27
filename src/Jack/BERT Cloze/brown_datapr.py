import pandas as p
import os


def prepare_data():
    raw_data = p.read_csv("src/Jack/Data/brown.csv")
    sentences = raw_data["raw_text"].str.split(expand=True)
    sentences = sentences.apply(lambda x: x.str.rsplit("/").str[0])
    clean_sentences = sentences.apply(lambda x: x.str.cat(sep=" "), axis = 1)
    entire_corpus = clean_sentences.str.cat(sep=" [SEP] ")
    return entire_corpus

z = prepare_data()

os.getcwd(__file__)
import inspect


with open("src/Jack/Data/brown_string.txt", "w") as text_file:
    text_file.write(z)

