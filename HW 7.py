import numpy as np
import pandas as pd
train=pd.read_csv("C:\\Users\\User\\Desktop\\Syracuse\\train.tsv", delimiter='\t')
y=train['Sentiment'].values
X=train['Phrase'].values
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
#%%
unique, counts = np.unique(y_train, return_counts=True)
print(np.asarray((unique, (counts/len(y_train))*100)))
#%%
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
unigram_count_vectorizer = CountVectorizer(encoding='latin-1', binary=False, min_df=5, stop_words='english')
unibigram_count_vectorizer = CountVectorizer(encoding='latin-1', binary=False, min_df=5, stop_words='english', ngram_range=(1, 2))
#unigram_count_vectorizer = TfidfVectorizer(encoding='latin-1', use_idf=True, min_df=5, stop_words='english')
#%%
X_train_vec = unigram_count_vectorizer.fit_transform(X_train)
X_train_vec2 = unibigram_count_vectorizer.fit_transform(X_train)

# check the content of a document vector
print(X_train_vec.shape)
print(X_train_vec[0].toarray())

# check the size of the constructed vocabulary
print(len(unigram_count_vectorizer.vocabulary_))
print(len(unibigram_count_vectorizer.vocabulary_))
#%%
X_test_vec = unigram_count_vectorizer.transform(X_test)
X_test_vec2 = unibigram_count_vectorizer.transform(X_test)

# print out #examples and #features in the test set
print(X_test_vec.shape)
#%%
from sklearn.naive_bayes import MultinomialNB

# initialize the MNB model
nb_clf= MultinomialNB()
nb_clf2= MultinomialNB()
# use the training data to train the MNB model
nb_clf.fit(X_train_vec,y_train)
nb_clf2.fit(X_train_vec2,y_train)
#%%
feature_ranks = sorted(zip(nb_clf.feature_log_prob_[0], unigram_count_vectorizer.get_feature_names()))
very_negative_features = feature_ranks[-10:]
for i in range(0, len(very_negative_features)):
    print(very_negative_features[i])

feature_ranks2 = sorted(zip(nb_clf.feature_log_prob_[4], unigram_count_vectorizer.get_feature_names()))
very_positive_features = feature_ranks2[-10:]
for i in range(0, len(very_positive_features)):
    print(very_positive_features[i])
#%%
nb_clf.score(X_test_vec,y_test)
nb_clf2.score(X_test_vec2,y_test)
#%%
from sklearn.metrics import confusion_matrix
y_pred = nb_clf.fit(X_train_vec, y_train).predict(X_test_vec)
cm=confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4])
print(cm)
from sklearn.metrics import classification_report
target_names = ['0','1','2','3','4']
print(classification_report(y_test, y_pred, target_names=target_names))

y_pred3 = nb_clf2.fit(X_train_vec2, y_train).predict(X_test_vec2)
cm3=confusion_matrix(y_test, y_pred3, labels=[0,1,2,3,4])
print(cm3)
print(classification_report(y_test, y_pred3, target_names=target_names))
#%%
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print(precision_score(y_test, y_pred, average=None))
print(recall_score(y_test, y_pred, average=None))


#%%
from sklearn.svm import LinearSVC

# initialize the LinearSVC model
svm_clf = LinearSVC(C=1)
svm_clf2 = LinearSVC(C=1)
# use the training data to train the model
svm_clf.fit(X_train_vec,y_train)
svm_clf2.fit(X_train_vec2,y_train)
#%%
feature_ranks3 = sorted(zip(svm_clf.coef_[0], unigram_count_vectorizer.get_feature_names()))
very_negative_10 = feature_ranks3[-10:]
print("Very negative words")
for i in range(0, len(very_negative_10)):
    print(very_negative_10[i])
print()

feature_ranks4 = sorted(zip(svm_clf.coef_[4], unigram_count_vectorizer.get_feature_names()))
very_positive_10 = feature_ranks4[-10:]
print("Very positive words")
for i in range(0, len(very_positive_10)):
    print(very_positive_10[i])
print() 
#%%
svm_clf.score(X_test_vec,y_test)
svm_clf2.score(X_test_vec2,y_test)

y_pred2 = svm_clf.predict(X_test_vec)
cm2=confusion_matrix(y_test, y_pred2, labels=[0,1,2,3,4])
print(cm2)
print()

target_names = ['0','1','2','3','4']
print(classification_report(y_test, y_pred2, target_names=target_names))

y_pred4 = svm_clf2.predict(X_test_vec2)
cm4=confusion_matrix(y_test, y_pred4, labels=[0,1,2,3,4])
print(cm4)
print(classification_report(y_test, y_pred4, target_names=target_names))
#%%
from sklearn.model_selection import cross_val_score
svm_clf3 = LinearSVC(C=0.5)
X_vec = unibigram_count_vectorizer.transform(X)
svm_clf3.fit(X_vec,y)
print(cross_val_score(svm_clf3, X_vec, y, cv=5))
