import pandas as pd

def build_model(train_df, test_df, model):
    df = pd.get_dummies(pd.concat([train_df[["Pclass", "Sex", "Age", "Fare", 'Embarked', 'Family', 'Title', 'Deck']], test_df[["Pclass", "Sex", "Age", "Fare", 'Embarked', 'Family', 'Title', 'Deck']]], ignore_index=1))
    x_train = df.iloc[:891]
    x_test = df.iloc[891:]
    y_train = train_df["Survived"]
    clf = model
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return y_pred


def export_to_csv(train_df, test_df, model):
    y_pred = build_model(train_df, test_df, model)
    output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': y_pred})
    output.to_csv('titanic_submission.csv', index=False)