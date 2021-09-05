import numpy as np


def find_title(substring_list, big_string):
    for substring in substring_list:
        if substring in big_string:
            return substring
        
    print(big_string)
    return np.nan

def replace_title(df):
    title = df["Title"]
    if title in ['Master', 'Major', 'Rev', 'Col', 'Capt', 'Don', 'Jonkheer']:
        return 'Mr'
    elif title in ['Ms', 'Mlle']:
        return 'Miss'
    elif title in ['Countess', 'MMe']:
        return 'Mrs'
    elif title=='Dr':
        if df['Sex']=='male':
            return 'Mr'
        else:
            return 'Mrs'
    return title

def clean_data(train_df, test_df):
    train_df["Family"] = train_df["SibSp"] + train_df["Parch"] + 1
    test_df["Family"] = test_df["SibSp"] + test_df["Parch"] + 1

    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                        'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                        'Don', 'Jonkheer']
    train_df["Title"] = train_df["Name"].map(lambda x: find_title(title_list, x))
    test_df["Title"] = test_df["Name"].map(lambda x: find_title(title_list, x))



    train_df["Title"] = train_df.apply(replace_title, axis=1)
    test_df["Title"] = test_df.apply(replace_title, axis=1)

    train_df["Embarked"] = train_df.Embarked.fillna('S')
    train_df["Deck"] = train_df.Cabin.apply(lambda x: x[0] if x is not np.nan else None)
    test_df["Deck"] = test_df.Cabin.apply(lambda x: x[0] if x is not np.nan else None)

    train_df.Age = train_df.groupby(["Family", "Pclass"]).Age.apply(lambda x: x.fillna(x.mean()))
    test_df.Age = test_df.groupby(["Family", "Pclass"]).Age.apply(lambda x: x.fillna(x.mean()))

    train_df['Age'][159] = 5
    train_df['Age'][180] = 7
    train_df['Age'][201] = 16
    train_df['Age'][324] = 19
    train_df['Age'][792] = 20
    train_df['Age'][846] = 18
    train_df['Age'][863] = 14

    train_df["Deck"]=train_df["Deck"].fillna("M")
    test_df["Deck"]=test_df["Deck"].fillna("M")

    test_df.Fare = test_df.groupby(["Family", "Pclass"]).Fare.apply(lambda x: x.fillna(x.mean())).isnull().sum()

    return train_df, test_df
