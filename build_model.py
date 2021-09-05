from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np


def predict_score(x, y, clf):
    x_train, x_test, y_train, y_test = train_test_split(x,y)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    ratio = sum(pred==y_test)/len(pred)
    print(clf.__class__.__name__, ' percentage: ', ratio)
    print(clf.__class__.__name__, "cross validation score: ", np.mean(cross_val_score(clf, x_train, y_train, cv=10)))


def apply_models(train_df):
    x = pd.get_dummies(train_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Ticket", 'Embarked', 'Family', 'Title', 'Deck']])
    y = train_df["Survived"]

    models = [RandomForestClassifier(), LogisticRegression(), SVC(), MLPClassifier()]

    for model in models:
        try:
            predict_score(x, y, model)
        except:
            print(model.__class__.__name__, " not possible")
