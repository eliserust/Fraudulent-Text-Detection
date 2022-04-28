# Fraudulent-Text-Detection
Final Project for Computational Linguistics class - classification of LIAR fraudulent text dataset using handful of NLP and ML methods

LIAR Dataset available for download here: https://paperswithcode.com/dataset/liar#:~:text=Fake%20News%20Detection-,LIAR%20is%20a%20publicly%20available%20dataset%20for%20fake%20news%20detection,fact%2Dchecking%20research%20as%20well.


Methodology:
1. Parse train, dev, test data --> isolate text data, labels, speaker, political affiliation of speaker, and context.
2. Pre-process text data --> Stemming, removal of stopwords, punctuation, tokenization, etc.
3. Convert to TFIdf Vectorized Form
4. Fit SVM (Linear Kernel), SVM (Polynomial Kernel), Multinomial NB, and DT models to train data
5. Test with dev data --> tweak parameters 
6. Test with test data --> can be edited in argparse (replace valid.tsv with test.tsv)
7. Test with text alone, and iteratively with speakers, subject, and party data
8. Construct Confusion Matrices, Heatmap
9. Mutual Information Analysis --> Clean and Vectorize data and calculate Mutual Information Classification Score
10. Identify most important features for classification


Key Packages (Not Included in Class) Import Instructions
1. from sklearn.feature_extraction.text import CountVectorizer
2. from sklearn.svm import LinearSVC # For SVM
3. from sklearn.svm import SVC # For SVM
4. from sklearn import tree # For DT
5. import seaborn as sn # For heatmap
