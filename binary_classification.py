import argparse
import numpy as np
import pandas as pd
import string
import warnings
import sys
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from Parsing import parse_tsv
from sklearn.svm import LinearSVC # New package
from sklearn.svm import SVC # New package
from sklearn import svm, datasets # New package

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
    return " ".join(toks_nostop)

# Main function
def main(train_data, dev_data):

    # Load texts and labels
    train_labels, train_texts, train_subjects, train_speakers, train_parties = parse_tsv(train_data)
    dev_labels, dev_texts, dev_subjects, dev_speakers, dev_parties = parse_tsv(dev_data)


    # Preprocess texts
    preproc_texts = [preprocess_text(text) for text in train_texts]

    # Build TFIDF vector feature matrix and fit to training data
    vectorizer = TfidfVectorizer(input = "content", lowercase = True, analyzer = "word", use_idf = True, min_df = 10)
    tfidf_vector = vectorizer.fit_transform(train_texts) # Ik we're never to .fit_transform not sure what the workaround is
    train_df = pd.DataFrame(tfidf_vector.toarray(), columns=vectorizer.get_feature_names())
    # Remove number columns??

    # TFIDF vector feature matrix for preprocessed data
    preproc_vectorizer = TfidfVectorizer(input = "content", lowercase = True, analyzer = "word", use_idf = True,
                                         min_df = 10, token_pattern = "\S+")
    tfidf_preproc = preproc_vectorizer.fit_transform(preproc_texts)
    preproc_train_df = pd.DataFrame(tfidf_preproc.toarray(), columns=preproc_vectorizer.get_feature_names())
    #print(preproc_train_df)

    # TFIDF vector feature matrix for dev data
    dev_vector = vectorizer.transform(dev_texts)
    dev_df = pd.DataFrame(dev_vector.toarray(), columns=vectorizer.get_feature_names())

    print(train_df)
    print(dev_df)
    print(train_labels)


    print(f"The training data has shape {train_df.shape} and dtype {type(train_df)}")
    print(f"The testing data has shape {dev_df.shape} and dtype {type(dev_df)}")
    print(f"The training labels have shape {np.shape(train_labels)} and dtype {type(train_labels)}")
    print(f"The testing labels has shape {np.shape(dev_labels)} and dtype {type(dev_labels)}")

    # Load true/false labels into vector y, mapped to 1, 0 for binary classification
    binary_labels = []
    for label in train_labels:
        if label == 'true':
            binary_labels.append(1)
        elif label == 'mostly-true':
            binary_labels.append(1)
        elif label == 'half-true':
            binary_labels.append(1)
        elif label == 'barely-true':
            binary_labels.append(0)
        elif label == 'false':
            binary_labels.append(0)
        elif label == 'pants-fire':
            binary_labels.append(0)

    binary_dev_labels = []
    for label in dev_labels:
        if label == 'true':
            binary_dev_labels.append(1)
        elif label == 'mostly-true':
            binary_dev_labels.append(1)
        elif label == 'half-true':
            binary_dev_labels.append(1)
        elif label == 'barely-true':
            binary_dev_labels.append(0)
        elif label == 'false':
            binary_dev_labels.append(0)
        elif label == 'pants-fire':
            binary_dev_labels.append(0)


    # Train an SVM model - linear kernel
    SVM_Model = LinearSVC(C = 1)  # Initialize SVM
    SVM_Model.fit(train_df, binary_labels)  # Train SVM with Training Data

    # Results
    print("SVM prediction:\n", SVM_Model.predict(dev_df))
    print("Actual:")
    print(binary_dev_labels)

    # RESULTS - Confusion Matrix, Accuracy/Precision/Recall/F1
    SVM_matrix = confusion_matrix(binary_dev_labels, SVM_Model.predict(dev_df))
    print("\nThe confusion matrix is:")
    print(SVM_matrix)
    print("\n\n")
    print(accuracy_score(binary_dev_labels, SVM_Model.predict(dev_df)))
    print(precision_score(binary_dev_labels, SVM_Model.predict(dev_df), average='micro'))
    print(recall_score(binary_dev_labels, SVM_Model.predict(dev_df), average = 'micro'))
    print(f1_score(binary_dev_labels, SVM_Model.predict(dev_df), average = 'micro'))


    # Train SVM model - polynomial kernel
    SVM_poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo')
    SVM_poly.fit(train_df, binary_labels)

    # Results
    print("SVM prediction:\n", SVM_poly.predict(dev_df))
    print("Actual:")
    print(binary_dev_labels)

    # Confusion Matrix
    SVM_matrix2 = confusion_matrix(binary_dev_labels, SVM_poly.predict(dev_df))
    print("\nThe confusion matrix is:")
    print(SVM_matrix2)
    print("\n\n")
    print(accuracy_score(binary_dev_labels, SVM_poly.predict(dev_df)))
    print(precision_score(binary_dev_labels, SVM_poly.predict(dev_df)))
    print(recall_score(binary_dev_labels, SVM_poly.predict(dev_df)))
    print(f1_score(binary_dev_labels, SVM_poly.predict(dev_df)))


    # Train a multinomial NB model
    NB_Model = MultinomialNB()
    NB_Model.fit(train_df, binary_labels)
    # Confusion Matrix
    NB_CM = confusion_matrix(binary_dev_labels, NB_Model.predict(dev_df))
    print("\nThe confusion matrix is:")
    print(NB_CM)
    print("\n\n")
    print(accuracy_score(binary_dev_labels, NB_Model.predict(dev_df)))
    print(precision_score(binary_dev_labels, NB_Model.predict(dev_df)))
    print(recall_score(binary_dev_labels, NB_Model.predict(dev_df)))
    print(f1_score(binary_dev_labels, NB_Model.predict(dev_df)))

    sys.exit()

    # Train a Bernoulli NB model
    # Need to convert to binary Xtrain and Xtest first
    train_df_binary = train_df != 0
    test_df_binary = test_df != 0

    bernoulli_model = BernoulliNB()

    # If time: Zero Rule as a baseline

    # Visualizations


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
