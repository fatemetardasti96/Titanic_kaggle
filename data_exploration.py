import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


def explor_training_data(train_df):
    print("five rows from train data: ", train_df.head())
    print("five rows from test data: ", train_df.head())

    survived_age = train_df.loc[train_df.Survived==1]["Age"]
    dead_age = train_df.loc[train_df.Survived==0]["Age"]
    plt.hist(survived_age)
    plt.title("age distribution of survived cases")
    plt.show()
    plt.figure()
    plt.hist(dead_age)
    plt.title("age distribution of dead cases")
    plt.show()


    women = train_df.loc[train_df.Sex=='female']["Survived"]
    print("survived women ratio: ", sum(women)/len(women)*100, "%")

    men = train_df.loc[train_df.Sex=='male']["Survived"]
    print("survived men ratio: ", sum(men)/len(men)*100, "%")


    survived_fare = train_df.loc[train_df.Survived == 1]["Fare"]
    dead_fare = train_df.loc[train_df.Survived == 0]["Fare"]
    print("average fare of people who survived: ", survived_fare.mean())
    print("average fare of people who did not survive: ", dead_fare.mean())


    print("unique values of SibSp: ", train_df.SibSp.unique())
    print("unique values of Cabin: ", train_df.Cabin.unique())
    print("unique values of Ticket: ", train_df.Ticket.unique())
    print("unique values of Embarked: ", train_df.Embarked.unique())

    train_df.Embarked.hist()

    print("unique values of Parch: ", train_df.Parch.unique())

    print("count nan values in each columns: ", train_df.isna().sum())

    print("training data correlation: ", pd.get_dummies(train_df).corr())

    px.imshow(train_df.corr())