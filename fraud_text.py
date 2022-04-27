import argparse
import numpy as np
import pandas as pd
import string
import warnings
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from Parsing import parse_tsv
from sklearn.svm import LinearSVC # New package
from sklearn.svm import SVC # New package
from sklearn import svm, datasets # New package
from sklearn import tree # New package
import graphviz

warnings.filterwarnings("ignore") # ignore warnings

# Main function
def main(train_data, dev_data):

    # Load texts and labels
    train_labels, train_texts, train_subjects, train_speakers, train_parties = parse_tsv(train_data)
    dev_labels, dev_texts, dev_subjects, dev_speakers, dev_parties = parse_tsv(dev_data)

    # Build TFIDF vector feature matrix and fit to training data
    vectorizer = TfidfVectorizer(input = "content", lowercase = True, analyzer = "word", min_df = 5)
    tfidf_vector = vectorizer.fit_transform(train_texts) # Ik we're never to .fit_transform not sure what the workaround is
    train_df = pd.DataFrame(tfidf_vector.toarray(), columns=vectorizer.get_feature_names())

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


    # Decision Tree
    DT_Model = tree.DecisionTreeClassifier(criterion = 'entropy',
                                           splitter = 'best',
                                           max_depth = 10,
                                           min_samples_split = 2,
                                           min_samples_leaf = 1)
    DT_Model.fit(train_df, train_labels)
    # Confusion Matrix
    DT_CM = confusion_matrix(dev_labels, DT_Model.predict(dev_df))
    print("\nThe confusion matrix for Decision Tree is:")
    print(DT_CM)
    print("\n\n")
    print(accuracy_score(dev_labels, DT_Model.predict(dev_df)))
    print(precision_score(dev_labels, DT_Model.predict(dev_df), average='macro'))
    print(recall_score(dev_labels, DT_Model.predict(dev_df), average='macro'))
    print(f1_score(dev_labels, DT_Model.predict(dev_df), average='macro'))

    tree.plot_tree(DT_Model)
    topics = ['true', 'half-true', 'mostly-true', 'barely-true', 'false', 'pants-fire']
    feature_names = train_df.columns
    Tree_Object = tree.export_graphviz(DT_Model, out_file=None,
                                       feature_names=feature_names,
                                       class_names=topics,
                                       filled=True, rounded=True,
                                       special_characters=True)

    graph = graphviz.Source(Tree_Object)
    graph.render("MyTree_entropy_small")

    sys.exit()

    # Train an SVM model - linear kernel
    SVM_Model = LinearSVC(C=1)  # Initialize SVM
    SVM_Model.fit(train_df, train_labels)  # Train SVM with Training Data

    # Results
    print("Linear Kernel SVM prediction:\n", SVM_Model.predict(dev_df))
    print("Actual:")
    print(dev_labels)

    # RESULTS - Confusion Matrix, Accuracy/Precision/Recall/F1
    SVM_matrix = confusion_matrix(dev_labels, SVM_Model.predict(dev_df))
    print("\nThe confusion matrix for SVM (Linear Kernel) is:")
    print(SVM_matrix)
    print("\n\n")
    print(accuracy_score(dev_labels, SVM_Model.predict(dev_df)))
    print(precision_score(dev_labels, SVM_Model.predict(dev_df), average='macro'))
    print(recall_score(dev_labels, SVM_Model.predict(dev_df), average='macro'))
    print(f1_score(dev_labels, SVM_Model.predict(dev_df), average='macro'))

    # Train SVM model - polynomial kernel
    SVM_poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo')
    SVM_poly.fit(train_df, train_labels)

    # Results
    print("SVM polynomial kernel prediction:\n", SVM_poly.predict(dev_df))
    print("Actual:")
    print(dev_labels)

    # Confusion Matrix
    SVM_matrix2 = confusion_matrix(dev_labels, SVM_poly.predict(dev_df))
    print("\nThe confusion matrix for SVM (polynomial kernel) is:")
    print(SVM_matrix2)
    print("\n\n")
    print(accuracy_score(dev_labels, SVM_poly.predict(dev_df)))
    print(precision_score(dev_labels, SVM_poly.predict(dev_df), average='macro'))
    print(recall_score(dev_labels, SVM_poly.predict(dev_df), average='macro'))
    print(f1_score(dev_labels, SVM_poly.predict(dev_df), average='macro'))

    # Train a multinomial NB model
    NB_Model = MultinomialNB()
    NB_Model.fit(train_df, train_labels)
    # Confusion Matrix
    NB_CM = confusion_matrix(dev_labels, NB_Model.predict(dev_df))
    print("\nThe confusion matrix for Multinomial NB is:")
    print(NB_CM)
    print("\n\n")
    print(accuracy_score(dev_labels, NB_Model.predict(dev_df)))
    print(precision_score(dev_labels, NB_Model.predict(dev_df), average='macro'))
    print(recall_score(dev_labels, NB_Model.predict(dev_df), average='macro'))
    print(f1_score(dev_labels, NB_Model.predict(dev_df), average='macro'))

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
