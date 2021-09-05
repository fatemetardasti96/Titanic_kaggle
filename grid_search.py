from inspect import Parameter
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split


def apply_grid_search(train_df, estimator):
    x = pd.get_dummies(train_df[["Pclass", "Sex", "Age", "Fare", "Ticket", 'Embarked', 'Family', 'Title', 'Deck']])
    y = train_df["Survived"]
    x_train, x_test, y_train, y_test = train_test_split(x,y)

    if estimator.__class__.__name__ == "RandomForestClassifier":
        parameters = {"max_depth": (70, 80), "n_jobs": (2, ), "min_samples_split": (6, 4), "min_samples_leaf": (1, 3, 5), "min_weight_fraction_leaf":(0, 0.3)}        

    if estimator.__class__.__name__ == "MLPClassifier":
        parameters = {"hidden_layer_sizes": (200, ), "batch_size": (64, 32)}

    clf = GridSearchCV(estimator, parameters)
    clf.fit(x_train, y_train)

    print("best estimator: ", clf.best_params_)
    print("best estimator score: ", clf.score(x_test, y_test))