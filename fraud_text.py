import argparse
import numpy as np
import pandas as pd
import string
import sys
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr
from Parsing import parse_tsv

# Text pre-processing
def preprocess_text(text):
    # CREDIT: Copied from sts_tfidf lab
    """Preprocess one sentence:
    tokenizes, lowercases, applies the Porter stemmer,
     removes punctuation tokens and stopwords.
     Returns a string of tokens joined by whitespace.
     """
    # Stemmer
    stemmer = PorterStemmer()
    stops = set(stopwords.words('english')) # Identify  stop words
    toks = word_tokenize(text) # Tokenize sentence
    toks_stemmed = [stemmer.stem(tok.lower()) for tok in toks] # STEM tokens
    toks_nopunc = [tok for tok in toks_stemmed if tok not in string.punctuation] # Remove punctuation
    toks_nostop = [tok for tok in toks_nopunc if tok not in stops] # Remove stopwords
    return " ".join(toks_nostop)

# Main function
def main(train_data, test_data):

    # Load texts and labels
    train_labels, train_texts, train_subjects, train_speakers, train_parties = parse_tsv(train_data)
    test_labels, test_texts, test_subjects, test_speakers, test_parties = parse_tsv(test_data)

    # Preprocess texts
    #preproc_texts = [preprocess_text(text) for text in train_texts]

    # Build TFIDF vector feature matrix and fit to data
    train_texts = list(train_texts)
    #print(train_texts)
    #sys.exit()

    vectorizer = TfidfVectorizer(input = "content", lowercase = True, analyzer = "word", use_idf = True, min_df = 10)
    tfidf_vector = vectorizer.fit(train_texts)
    print("Checking the vocabulary: ")
    print(tfidf_vector.get_feature_names())

    # Load true/false labels into vector y, mapped to 0 and 1

    # Train an SVM model

    # Train a multinomial NB model

    # Train a Bernoulli NB model

    # Train a logistic regression model

    # Fit the zero rule

    # Report accuracy scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FraudulentTextDetection_Final')
    parser.add_argument('--train', type=str, default="liar_dataset/train.tsv",
                        help='path to training set')
    parser.add_argument('--dev', type=str, default="liar_dataset/valid.tsv",
                        help='path to dev set')
    #parser.add_argument('--seed', type=int, default=7,
    #                    help='random seed for dataset split')
    args = parser.parse_args()

    main(args.train, args.dev)
