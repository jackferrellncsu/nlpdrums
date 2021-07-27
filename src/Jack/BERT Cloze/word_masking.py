import csv
import pandas as p

raw_data = p.read_csv("src/Jack/Data/brown.csv")

sentences = raw_data["raw_text"]

sentences.apply(string.split)
