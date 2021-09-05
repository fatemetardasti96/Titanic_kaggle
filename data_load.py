import numpy as np 
import pandas as pd


def load_data():
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    return train_df, test_df