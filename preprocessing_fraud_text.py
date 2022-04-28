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
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from Parsing import parse_tsv
from sklearn import svm, datasets # New package
from sklearn.svm import LinearSVC # New package
from sklearn.svm import SVC # New package
from sklearn import tree # New package
import seaborn as sn
import matplotlib.pyplot as plt

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
    topics = ['true', 'half-true', 'mostly-true', 'barely-true', 'false', 'pants-fire']

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

    print(preproc_train_df)
    print(preproc_dev_df)
    print(train_labels)

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


    print(f"The training data has shape {preproc_train_df.shape} and dtype {type(preproc_train_df)}")
    print(f"The testing data has shape {preproc_dev_df.shape} and dtype {type(preproc_dev_df)}")
    print(f"The training labels have shape {np.shape(train_labels)} and dtype {type(train_labels)}")
    print(f"The testing labels has shape {np.shape(dev_labels)} and dtype {type(dev_labels)}")


    #### SVM model - linear kernel
    SVM_Model = LinearSVC(C = 1)  # Initialize SVM
    SVM_Model.fit(preproc_train_df, train_labels)  # Train SVM with Training Data

    # Results
    print("Linear Kernel SVM prediction:\n", SVM_Model.predict(preproc_dev_df))
    print("Actual:")
    print(dev_labels)

    # RESULTS - Confusion Matrix, Accuracy/Precision/Recall/F1
    SVM_matrix = confusion_matrix(dev_labels, SVM_Model.predict(preproc_dev_df))
    print("\nThe confusion matrix for SVM (Linear Kernel) is:")
    print(SVM_matrix)
    print("\n\n")
    print(accuracy_score(dev_labels, SVM_Model.predict(preproc_dev_df)))
    print(precision_score(dev_labels, SVM_Model.predict(preproc_dev_df), average='macro'))
    print(recall_score(dev_labels, SVM_Model.predict(preproc_dev_df), average = 'macro'))
    print(f1_score(dev_labels, SVM_Model.predict(preproc_dev_df), average = 'macro'))


    #### SVM model - polynomial kernel
    SVM_poly = SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo')
    SVM_poly.fit(preproc_train_df, train_labels)

    # Results
    print("SVM polynomial kernel prediction:\n", SVM_poly.predict(preproc_dev_df))
    print("Actual:")
    print(dev_labels)

    # Confusion Matrix
    SVM_matrix2 = confusion_matrix(dev_labels, SVM_poly.predict(preproc_dev_df), labels = topics)
    print("\nThe confusion matrix for SVM (polynomial kernel) is:")
    print(SVM_matrix2)
    print("\n\n")
    print(accuracy_score(dev_labels, SVM_poly.predict(preproc_dev_df)))
    print(precision_score(dev_labels, SVM_poly.predict(preproc_dev_df), average='macro'))
    print(recall_score(dev_labels, SVM_poly.predict(preproc_dev_df), average='macro'))
    print(f1_score(dev_labels, SVM_poly.predict(preproc_dev_df), average='macro'))


    #### Multinomial NB model
    NB_Model = MultinomialNB()
    NB_Model.fit(preproc_train_df, train_labels)
    # Confusion Matrix
    NB_CM = confusion_matrix(dev_labels, NB_Model.predict(preproc_dev_df))
    print("\nThe confusion matrix for Multinomial NB is:")
    print(NB_CM)
    print("\n\n")
    print(accuracy_score(dev_labels, NB_Model.predict(preproc_dev_df)))
    print(precision_score(dev_labels, NB_Model.predict(preproc_dev_df), average = 'macro'))
    print(recall_score(dev_labels, NB_Model.predict(preproc_dev_df), average = 'macro'))
    print(f1_score(dev_labels, NB_Model.predict(preproc_dev_df), average = 'macro'))

    #### Decision Tree
    DT_Model = tree.DecisionTreeClassifier(criterion='entropy',
                                           splitter='best',
                                           max_depth=10,
                                           min_samples_split=2,
                                           min_samples_leaf=1)
    DT_Model.fit(preproc_train_df, train_labels)
    # Confusion Matrix
    DT_CM = confusion_matrix(dev_labels, DT_Model.predict(preproc_dev_df))
    print("\nThe confusion matrix for Decision Tree is:")
    print(DT_CM)
    print("\n\n")
    print(accuracy_score(dev_labels, DT_Model.predict(preproc_dev_df)))
    print(precision_score(dev_labels, DT_Model.predict(preproc_dev_df), average='macro'))
    print(recall_score(dev_labels, DT_Model.predict(preproc_dev_df), average='macro'))
    print(f1_score(dev_labels, DT_Model.predict(preproc_dev_df), average='macro'))

    #### Visualizations
    cm = pd.DataFrame(SVM_matrix2, index=[i for i in topics],
                      columns=[i for i in topics])
    plt.figure(figsize=(10, 7))
    svm = sn.heatmap(cm, annot=True, cmap="Blues")
    figure = svm.get_figure()
    figure.savefig('svm_conf.png', dpi=400)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FraudulentTextDetection_Final')
    parser.add_argument('--train', type=str, default="liar_dataset/train.tsv",
                        help='path to training set')
    parser.add_argument('--dev', type=str, default="liar_dataset/test.tsv",
                        help='path to dev set')
    args = parser.parse_args()

    main(args.train, args.dev)
