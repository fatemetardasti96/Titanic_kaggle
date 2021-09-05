# Titanic_kaggle

data_load.py: load training and test data

data_exploration.py: explore data to get more idea about how data looks like and what are the releationship between different parameters and survived column. **Optional**

data_cleaning.py: fill empty cells and apply feature engineering to get Family and Title and Deck columns

build_model.py: build preliminary models based on training data to get a first impression  which model among `RandomForestClassifier(), LogisticRegression(),
SVC(), MLPClassifier()` works the best. **Optional**

grid_search.py: apply grid search for RFC and MLP. **Optional**

build_export_output_model.py: build final model based on grid search result and write the output result into csv

#RUN
`python run.py 0 0 1`

`
explor_data = argv[1]
build_model = argv[2]
should_apply_grid_search = argv[3]
`
result after applying feature engineering: 
`RFC(max_depth=80, n_jobs=2, min_samples_split=4): 75.358%
`
