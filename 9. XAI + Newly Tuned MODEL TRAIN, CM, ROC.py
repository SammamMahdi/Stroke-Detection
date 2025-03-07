import pandas as pd
from holoviews.plotting.bokeh.styles import font_size
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, auc, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import seaborn as sns
import time
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
import pickle
# Load the dataset
df = pd.read_csv("COMBO2.csv")

# Split the dataset into training and testing sets

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
df = pd.read_csv("COMBO2.csv")
cb = CatBoostClassifier(**hyperparameters_CB)
xgb = XGBClassifier(**hyperparameters_XGB, )
rf = RandomForestClassifier(**hyperparameters_RFC, random_state=150)
# Initialize and train the RFC, SVM, and XGBoost models
models = [
    ("XGB", xgb, "Blues"),
    ("RF", rf, "Purples"),
    ("CB", cb, "Reds"),
    ("MVC", VotingClassifier(estimators=[
        ("cb", cb),
        ('xgb', xgb),
        ('rf', rf),
    ], voting='hard'), "Greens")
]
# Only keeping Top 10 regions
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
print(len(drop_columns))
X = df.drop(drop_columns, axis=1)
y = df["is_stroke_face"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=150)

for name, model, color in models:
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf = confusion_matrix(y_test, y_pred)

    plt.title(f"{name} | Acc: {accuracy * 100:.2f}% | Time: {end - start:.2f} sec")

    labels = ["Non-Stroke", "Stroke", "Non-Stroke", "Stroke"]
    group_percentages = ["{0:.2%}".format(value) for value in conf.flatten() / np.sum(conf)]
    categories = ["Non_stroke", "Stroke"]
    labels = np.asarray(labels).reshape(2, 2)
    group_percentages = np.asarray(group_percentages).reshape(2, 2)
    sns.heatmap(conf, annot=group_percentages, fmt="", cmap=color, xticklabels=categories, yticklabels=categories,
                annot_kws={"size": 30})  # Adjust font size here
    plt.rcParams['savefig.dpi'] = 1000  # image quality
    plt.show()

# Plot ROC curve for each model
plt.figure(figsize=(20, 20))
#
acc_score = {name: 0 for name, model, color in models}
# while acc_score["Voting"] <= acc_score["CatBoost"]:
for name, model, color in models:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    acc_score[name] = accuracy
    print(f'Accuracy for {name}: {accuracy * 100:.2f}%')
    print(classification_report(y_test, y_pred, zero_division=1))
    if name != "MVC":
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.rcParams['savefig.dpi'] = 1000  # image quality
        plt.plot(fpr, tpr, label=f'{name}| AUC : {roc_auc:.2f} Acc : {accuracy * 100:.2f}%')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=22)  # Adjusted font size for x-axis label
plt.ylabel('True Positive Rate', fontsize=22)  # Adjusted font size for y-axis label
plt.title('Receiver Operating Characteristic', fontsize=24)  # Adjusted font size for title
plt.xticks(fontsize=20)  # Adjusted font size for x-tick labels
plt.yticks(fontsize=20)  # Adjusted font size for y-tick labels
plt.legend(loc="lower right", fontsize=20)  # Adjusted legend font size
plt.show()
