#Import Libraries
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#load iris  data
iris = datasets.load_iris()
#X Data
X = iris.data[:, 1:3]
#y Data
y = iris.target

#Classification models
clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
clf4 = LinearDiscriminantAnalysis(n_components=3 ,solver='svd')


for clf, label in zip([clf1, clf2, clf3, clf4], ['Logistic Regression', 'Random Forest ', 'naive Bayes ', 'Linear Discriminant Analysis( ']): 
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))