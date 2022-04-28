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
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from Parsing import parse_tsv
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier


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
def main(train_data):

    # Load texts and labels
    train_labels, train_texts, train_subjects, train_speakers, train_parties = parse_tsv(train_data)
    preproc_train_texts = [preprocess_text(text) for text in train_texts]

    # TFIDF vector feature matrix for preprocessed data
    preproc_vectorizer = TfidfVectorizer(input = "content", lowercase = True, analyzer = "word", use_idf = True,
                                         min_df = 5, token_pattern = "\S+", ngram_range = (1,2))
    tfidf_preproc = preproc_vectorizer.fit_transform(preproc_train_texts)
    preproc_train_df = pd.DataFrame(tfidf_preproc.toarray(), columns=preproc_vectorizer.get_feature_names())


    #### Vectorize subjects, speakers, parties data
    count_vec = CountVectorizer()

    train_speakers_vec = count_vec.fit_transform(train_speakers).toarray()
    speakers_df = pd.DataFrame(train_speakers_vec)
    speakers_df.columns = count_vec.get_feature_names()


    train_subjects_vec = count_vec.fit_transform(train_subjects).toarray()
    subjects_df = pd.DataFrame(train_subjects_vec)
    subjects_df.columns = count_vec.get_feature_names()

    train_parties_vec = count_vec.fit_transform(train_parties).toarray()
    parties_df = pd.DataFrame(train_parties_vec)
    parties_df.columns = count_vec.get_feature_names()


    ###### Mutual information
    # Speakers
    speakers_mi_score = MIC(speakers_df, train_labels)
    indices = np.argpartition(speakers_mi_score, -10)[-10:].tolist()
    top_texts = speakers_df.iloc[:, indices]
    text_features = top_texts.columns.values.tolist()
    print(text_features)

    # Parties
    parties_mi_score = MIC(parties_df, train_labels)
    indices = np.argpartition(parties_mi_score, -10)[-10:].tolist()
    top_texts = parties_df.iloc[:, indices]
    text_features = top_texts.columns.values.tolist()
    print(text_features)

    # Text
    text_mi_score = MIC(preproc_train_df, train_labels)
    indices = np.argpartition(text_mi_score, -10)[-10:].tolist()
    top_texts = preproc_train_df.iloc[:, indices]
    text_features = top_texts.columns.values.tolist()
    print(text_features)

    # Subjects
    subjects_mi_score = MIC(subjects_df, train_labels)
    indices = np.argpartition(subjects_mi_score, -10)[-10:].tolist()
    top_texts = subjects_df.iloc[:, indices]
    text_features = top_texts.columns.values.tolist()
    print(text_features)

    # Print highest MIC scores for each
    print(np.sort(-text_mi_score))
    print(np.sort(-subjects_mi_score))
    print(np.sort(-speakers_mi_score))
    print(np.sort(-parties_mi_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FraudulentTextDetection_Final')
    parser.add_argument('--train', type=str, default="liar_dataset/train.tsv",
                        help='path to training set')
    args = parser.parse_args()

    main(args.train)
