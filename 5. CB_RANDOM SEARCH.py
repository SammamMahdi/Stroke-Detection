import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import randint, uniform
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score


df = pd.read_csv('Sleep_Stage_Combo.csv')

drop_columns = ['SubNo', "SegNo", "Class"]

X = df.drop(drop_columns, axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)
# hyperparameters = {
#     'bagging_temperature': 0.9422017556848528,
#     'border_count': 14,
#     'depth': 5,
#     'iterations': 786,
#     'l2_leaf_reg': 5,
#     'learning_rate': 0.0792681476866447,
#     'random_strength': 0.24102546602601171
# }
# Initialize and train the CatBoost Classifier


model = CatBoostClassifier()
model.fit(X_train, y_train)
# Parameter distribution
catboost_param_dist = {
    'depth': randint(4, 10),
    'learning_rate': uniform(0.01, 0.3),
    'iterations': randint(100, 1000),
    'l2_leaf_reg': randint(1, 10),
    'border_count': randint(1, 255),
    'bagging_temperature': uniform(0.0, 1.0),
    'random_strength': uniform(0.0, 1.0),
    'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS', 'Poisson', None],
    'subsample': uniform(0.5, 1.0),
    'colsample_bylevel': uniform(0.5, 1.0),
    'scale_pos_weight': uniform(0.5, 2.0),
    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide', None],
    'min_data_in_leaf': randint(1, 20),
    'one_hot_max_size': randint(2, 10),
    'max_bin': randint(200, 300),
    'od_type': ['IncToDec', 'Iter', None],
    'od_wait': randint(10, 50)
}
random_search_cb = RandomizedSearchCV(estimator=model,
                                      param_distributions=catboost_param_dist,
                                      cv=5,
                                      verbose=2,
                                      random_state=42)
# # Fit the model
random_search_cb.fit(X_train, y_train)
# Evaluate the model
random_search_cb_score = random_search_cb.score(X_test, y_test)
best_parameters = random_search_cb.best_params_
best_score = random_search_cb.best_score_
print(f"Best Parameters: {best_parameters}")
print(f"Best Score: {best_score}")
# Evaluate and predict
# predictions = model.predict(X_test)
# accuracy = accuracy_score(y_test, predictions)
# print(accuracy)
