from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from catboost import CatBoostClassifier

# Split the data into training and testing sets
df = pd.read_csv("COMBO2.csv")
drop_columns = ['Filename', 'is_stroke_face', 'rightEyeLower1_130', 'rightEyeLower1_25', 'rightEyeLower1_110',
                'rightEyeLower1_24', 'rightEyeLower1_23', 'rightEyeLower1_22', 'rightEyeLower1_26',
                'rightEyeLower1_112', 'rightEyeLower1_243', 'rightEyeUpper0_246', 'rightEyeUpper0_161',
                'rightEyeUpper0_160', 'rightEyeUpper0_159', 'rightEyeUpper0_158', 'rightEyeUpper0_157',
                'rightEyeUpper0_173', 'rightEyeLower0_33', 'rightEyeLower0_7', 'rightEyeLower0_163',
                'rightEyeLower0_144', 'rightEyeLower0_145', 'rightEyeLower0_153', 'rightEyeLower0_154',
                'rightEyeLower0_155', 'rightEyeLower0_133', 'leftEyeLower3_372', 'leftEyeLower3_340',
                'leftEyeLower3_346', 'leftEyeLower3_347', 'leftEyeLower3_348', 'leftEyeLower3_349', 'leftEyeLower3_350',
                'leftEyeLower3_357', 'leftEyeLower3_465', 'rightEyeLower2_226', 'rightEyeLower2_31',
                'rightEyeLower2_228', 'rightEyeLower2_229', 'rightEyeLower2_230', 'rightEyeLower2_231',
                'rightEyeLower2_232', 'rightEyeLower2_233', 'rightEyeLower2_244', 'rightEyeUpper2_113',
                'rightEyeUpper2_225', 'rightEyeUpper2_224', 'rightEyeUpper2_223', 'rightEyeUpper2_222',
                'rightEyeUpper2_221', 'rightEyeUpper2_189', 'leftEyeUpper1_467', 'leftEyeUpper1_260',
                'leftEyeUpper1_259', 'leftEyeUpper1_257', 'leftEyeUpper1_258', 'leftEyeUpper1_286', 'leftEyeUpper1_414',
                'leftEyeLower2_446', 'leftEyeLower2_261', 'leftEyeLower2_448', 'leftEyeLower2_449', 'leftEyeLower2_450',
                'leftEyeLower2_451', 'leftEyeLower2_452', 'leftEyeLower2_453', 'leftEyeLower2_464', 'leftEyeLower1_359',
                'leftEyeLower1_255', 'leftEyeLower1_339', 'leftEyeLower1_254', 'leftEyeLower1_253', 'leftEyeLower1_252',
                'leftEyeLower1_256', 'leftEyeLower1_341', 'leftEyeLower1_463', 'leftEyeUpper2_342', 'leftEyeUpper2_445',
                'leftEyeUpper2_444', 'leftEyeUpper2_443', 'leftEyeUpper2_442', 'leftEyeUpper2_441', 'leftEyeUpper2_413',
                'rightEyebrowLower_35', 'rightEyebrowLower_124', 'rightEyebrowLower_46', 'rightEyebrowLower_53',
                'rightEyebrowLower_52', 'rightEyebrowLower_65', 'leftEyebrowLower_265', 'leftEyebrowLower_353',
                'leftEyebrowLower_276', 'leftEyebrowLower_283', 'leftEyebrowLower_282', 'leftEyebrowLower_295',
                'rightEyeUpper1_247', 'rightEyeUpper1_30', 'rightEyeUpper1_29', 'rightEyeUpper1_27',
                'rightEyeUpper1_28', 'rightEyeUpper1_56', 'rightEyeUpper1_190', 'leftEyeUpper0_466',
                'leftEyeUpper0_388', 'leftEyeUpper0_387', 'leftEyeUpper0_386', 'leftEyeUpper0_385', 'leftEyeUpper0_384',
                'leftEyeUpper0_398', 'noseBottom_2', 'midwayBetweenEyes_168', 'noseRightCorner_98',
                'noseLeftCorner_327']

X = df.drop(drop_columns, axis=1)
y = df["is_stroke_face"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=150)

hyperparameters_RFC = {"n_estimators": 400, "max_depth": 100, "min_samples_split": 10, "min_samples_leaf": 4,
                       "max_features": 0.25, "bootstrap": False, "criterion": "gini"}  # new hyperparameters for top 10

hyperparameters_XGB = {'max_depth': 7, 'min_child_weight': 2, 'learning_rate': 0.2, 'subsample': 0.8,
                       'colsample_bytree': 1.0, 'gamma': 0.3, 'n_estimators': 400, 'use_label_encoder': False,
                       'eval_metric': 'rmse', 'objective': 'binary:logistic'}  # new hyperparameters for top 10

hyperparameters_CB = {'bagging_temperature': 0.8607305832563434, 'bootstrap_type': 'MVS',
                      'colsample_bylevel': 0.917411003148779,
                      'depth': 8, 'grow_policy': 'SymmetricTree', 'iterations': 918, 'l2_leaf_reg': 8,
                      'learning_rate': 0.29287291117375575, 'max_bin': 231, 'min_data_in_leaf': 9, 'od_type': 'Iter',
                      'od_wait': 21, 'one_hot_max_size': 7, 'random_strength': 0.6963042728397884,
                      'scale_pos_weight': 1.924541179848884, 'subsample': 0.6480869299533999}
# Models
models = {
    'XGBoost': XGBClassifier(**hyperparameters_XGB),  # Set eval_metric to 'logloss' to monitor log loss
    'CB': CatBoostClassifier(**hyperparameters_CB),
    'Random Forest': RandomForestClassifier(**hyperparameters_RFC, random_state=150),
}

# Train and evaluate each model
for model_name, model in models.items():
    if model_name == 'XGBoost':  # For XGBoost, set eval_metric in constructor
        model.set_params(eval_metric='logloss')
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    else:
        model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    confusion_mat = confusion_matrix(y_test, predictions)

    # Output
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Confusion Matrix:\n{confusion_mat}")

    # For XGBoost, plot the training loss
    # if model_name == 'XGBoost':
    #     results = model.evals_result()
    #     train_errors = results['validation_0']['logloss']
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(train_errors, label='Training Loss')
    #     plt.xlabel('Number of Iterations')
    #     plt.ylabel('Log Loss')
    #     plt.title('Training Loss of XGBoost Classifier')
    #     plt.legend()
    #     plt.show()
    #     # For CatBoost, plot the training loss
    # if model_name == 'CB':
    #     results = model.get_evals_result()
    #     train_errors = results['learn']['Logloss']
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(train_errors, label='Training Loss')
    #     plt.xlabel('Number of Iterations')
    #     plt.ylabel('Log Loss')
    #     plt.title('Training Loss of CatBoost Classifier')
    #     plt.legend()
    #     plt.show()

    # For RandomForest, manually compute misclassification rate on the training set during training
    if model_name == 'Random Forest':
        from sklearn.metrics import accuracy_score

        n_estimators_range = range(10, 5001, 10)  # Example: from 10 to 400 trees, in steps of 10
        train_errors = []  # Store training errors
        for n_estimators in n_estimators_range:
            model.set_params(n_estimators=n_estimators)
            model.fit(X_train, y_train)
            train_predictions = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_predictions)
            train_error = 1 - train_accuracy  # Misclassification rate
            train_errors.append(train_error)
            print(f"Trained with {n_estimators} trees: Training Error = {train_error}")

        # Plotting the training error as a function of the number of trees
        plt.figure(figsize=(8, 6))
        plt.plot(n_estimators_range, train_errors, label='Training Error')
        plt.xlabel('Number of Trees')
        plt.ylabel('Training Error (Misclassification Rate)')
        plt.title('Training Error of RandomForest Classifier')
        plt.legend()
        plt.show()
