import numpy as np
import pandas as pd
import string
import system
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


## Define function to read in and convert .tsv files into .csv files
def parse_tsv(data_file):
    """
    Reads a tab-separated tsv file and returns
    texts: list of texts (sentences)
    labels: list of labels (fake or real news)
    """
    texts = []
    labels = []
    context = []

    with open(data_file, 'r') as dd:
        for line in dd:
            fields = line.strip().split("\t")
            labels.append(float(fields[4]))
            t1 = fields[5].lower()
            t2 = fields[6].lower()
            texts.append((t1, t2))
    return texts, labels

data = pd.read_csv('liar_dataset/train.tsv', sep='\t')
parse_tsv()