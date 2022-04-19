import pandas as pd
import sys


## Define function to read in and convert .tsv files into .csv files
def parse_tsv(data_file):
    """
    Reads a tab-separated tsv file and returns
    texts: list of texts (sentences)
    labels: list of labels (fake or real news)
    """
    labels = []
    texts = []
    subjects = []
    speakers = []
    parties = []

    with open(data_file, 'r') as dd:
        for line in dd:
            fields = line.strip().split("\t")
            labels.append(fields[1])
            texts.append(fields[2])
            subjects.append(fields[3])
            speakers.append(fields[4])
            parties.append(fields[7])

    return labels, texts, subjects, speakers, parties

    #with open(data_file) as dd:
    #    data = pd.read_csv(dd, sep='\t')
    #    # Append data to individual lists
    #    labels.append(data[data.columns[1]])
    #    texts.append(data[data.columns[2]])
    #    subjects.append(data[data.columns[3]])
    #    speakers.append(data[data.columns[4]])
    #    parties.append(data[data.columns[7]])
    #return labels, texts, subjects, speakers, parties

#labels, texts, subjects, speakers, parties = parse_tsv('liar_dataset/train.tsv')