import argparse
import numpy as np
import pandas as pd
import string
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Text pre-processing


# Main function
def main(datafile):
    pass

    # Load texts and labels

    # Build TFIDF vector feature matrix

    # Load true/false labels into vector y, mapped to 0 and 1

    # Shuffle, split, assign 75% to train, 25% to test

    # Train an SVM model

    # Train a multinomial NB model

    # Train a Bernoulli NB model

    # Train a logistic regression model

    # Fit the zero rule

    # Report accuracy scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FraudulentTextDetection_Final')
    parser.add_argument('--path', type=str, default="train.tsv",
                        help='path to training set')
    parser.add_argument('--function_words_path', type=str, default="ewl_function_words.txt",
                        help='path to the list of words to use as features')
    parser.add_argument('--seed', type=int, default=7,
                        help='random seed for dataset split')
    args = parser.parse_args()

    main(args.path, args.function_words_path, args.seed)
