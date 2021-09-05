import sys

from data_load import load_data
from data_exploration import explor_training_data
from data_cleaning import clean_data
from build_model import apply_models
from grid_search import apply_grid_search
from build_export_output_model import build_model, export_to_csv
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neural_network import MLPClassifier as MLP

if __name__ == '__main__':
        
    explor_data = int(sys.argv[1])
    build_model = int(sys.argv[2])
    should_apply_grid_search = int(sys.argv[3])

    print("load data")
    train_df, test_df = load_data()

    if explor_data:
        print("explore data")
        explor_training_data(train_df)

    print("clean data")
    train_df, test_df = clean_data(train_df, test_df)

    if build_model:
        print("build model")
        apply_models(train_df)
    
    
    if should_apply_grid_search:
        print("apply grid search")
        apply_grid_search(train_df, RFC())

    print("export output to csv")
    export_to_csv(train_df, test_df, RFC(max_depth=80, n_jobs=2, min_samples_split=4))