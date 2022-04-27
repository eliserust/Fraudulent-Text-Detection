import argparse
import numpy as np
import pandas as pd
import string
import warnings
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC # New package
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from Parsing import parse_tsv
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif as MIC


warnings.filterwarnings("ignore") # ignore warnings

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
    # Replace numbers with #
    for tok in toks_nostop:
        if type(tok) == int:
            tok = '#'
    return " ".join(toks_nostop)

# Main function
def main(train_data, dev_data):

    # Load texts and labels
    train_labels, train_texts, train_subjects, train_speakers, train_parties = parse_tsv(train_data)
    dev_labels, dev_texts, dev_subjects, dev_speakers, dev_parties = parse_tsv(dev_data)

    # Preprocess texts
    preproc_train_texts = [preprocess_text(text) for text in train_texts]
    preproc_dev_texts = [preprocess_text(text) for text in dev_texts]


    # TFIDF vector feature matrix for preprocessed data
    preproc_vectorizer = TfidfVectorizer(input = "content", lowercase = True, analyzer = "word", use_idf = True,
                                         min_df = 5, token_pattern = "\S+", ngram_range = (1,2))
    tfidf_preproc = preproc_vectorizer.fit_transform(preproc_train_texts)
    preproc_train_df = pd.DataFrame(tfidf_preproc.toarray(), columns=preproc_vectorizer.get_feature_names())

    # TFIDF vector feature matrix for dev data
    dev_vector = preproc_vectorizer.transform(preproc_dev_texts)
    preproc_dev_df = pd.DataFrame(dev_vector.toarray(), columns=preproc_vectorizer.get_feature_names())


    #### Add in External Data (speakers, subjects, parties) to both train and dev
    count_vec = CountVectorizer(analyzer = lambda x: x)
    train_speakers_vec = count_vec.fit_transform(train_speakers).toarray()
    train_subjects_vec = count_vec.transform(train_subjects).toarray()
    train_parties_vec = count_vec.transform(train_parties).toarray()
    dev_speakers_vec = count_vec.transform(dev_speakers).toarray()
    dev_subjects_vec = count_vec.transform(dev_subjects).toarray()
    dev_parties_vec = count_vec.transform(dev_parties).toarray()

    # Concatenate one-hot vectors to training data, then to dev data
    preproc_train_df = pd.concat([preproc_train_df, pd.DataFrame(train_speakers_vec)], axis = 1)
    preproc_train_df = pd.concat([preproc_train_df, pd.DataFrame(train_subjects_vec)], axis=1)
    preproc_train_df = pd.concat([preproc_train_df, pd.DataFrame(train_parties_vec)], axis=1)

    preproc_dev_df = pd.concat([preproc_dev_df, pd.DataFrame(dev_speakers_vec)], axis = 1)
    preproc_dev_df = pd.concat([preproc_dev_df, pd.DataFrame(dev_subjects_vec)], axis=1)
    preproc_dev_df = pd.concat([preproc_dev_df, pd.DataFrame(dev_parties_vec)], axis=1)


    ###### Mutual information
    mi_score = MIC(preproc_train_df, train_labels)
    print(mi_score)
    sys.exit()



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
