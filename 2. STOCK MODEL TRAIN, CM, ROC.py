import pandas as pd
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

# Load the dataset
df = pd.read_csv("COMBO2.csv")
cb = CatBoostClassifier()
xgb = XGBClassifier()
rf = RandomForestClassifier()
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
drop_columns = ['Filename', "is_stroke_face"]
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
