import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier

FACE_INDEXES = {
    "lipsUpperOuter": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    "lipsLowerOuter": [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    "lipsUpperInner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    "lipsLowerInner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
    "rightEyeUpper0": [246, 161, 160, 159, 158, 157, 173],
    "rightEyeLower0": [33, 7, 163, 144, 145, 153, 154, 155, 133],
    "rightEyeUpper1": [247, 30, 29, 27, 28, 56, 190],
    "rightEyeLower1": [130, 25, 110, 24, 23, 22, 26, 112, 243],
    "rightEyeUpper2": [113, 225, 224, 223, 222, 221, 189],
    "rightEyeLower2": [226, 31, 228, 229, 230, 231, 232, 233, 244],
    "rightEyeLower3": [143, 111, 117, 118, 119, 120, 121, 128, 245],
    "rightEyebrowUpper": [156, 70, 63, 105, 66, 107, 55, 193],
    "rightEyebrowLower": [35, 124, 46, 53, 52, 65],
    "leftEyeUpper0": [466, 388, 387, 386, 385, 384, 398],
    "leftEyeLower0": [263, 249, 390, 373, 374, 380, 381, 382, 362],
    "leftEyeUpper1": [467, 260, 259, 257, 258, 286, 414],
    "leftEyeLower1": [359, 255, 339, 254, 253, 252, 256, 341, 463],
    "leftEyeUpper2": [342, 445, 444, 443, 442, 441, 413],
    "leftEyeLower2": [446, 261, 448, 449, 450, 451, 452, 453, 464],
    "leftEyeLower3": [372, 340, 346, 347, 348, 349, 350, 357, 465],
    "leftEyebrowUpper": [383, 300, 293, 334, 296, 336, 285, 417],
    "leftEyebrowLower": [265, 353, 276, 283, 282, 295],
    "midwayBetweenEyes": [168],
    "noseTip": [1],
    "noseBottom": [2],
    "noseRightCorner": [98],
    "noseLeftCorner": [327],
    "rightCheek": [205, 137, 123, 50, 203, 177, 147, 187, 207, 216, 215, 213, 192, 214, 212, 138, 135, 210, 169],
    "leftCheek": [425, 352, 280, 330, 266, 423, 426, 427, 411, 376, 436, 416, 432, 434, 422, 430, 364, 394, 371]
}

ranked_regions = ['rightCheek', 'leftCheek', 'lipsUpperOuter', 'lipsUpperInner', 'lipsLowerInner', 'lipsLowerOuter',
                  'rightEyebrowUpper', 'rightEyeLower3', 'leftEyeLower0', 'leftEyebrowUpper', 'rightEyeLower1',
                  'rightEyeUpper0', 'rightEyeLower0', 'leftEyeLower3', 'rightEyeLower2', 'rightEyeUpper2',
                  'leftEyeUpper1', 'leftEyeLower2', 'leftEyeLower1', 'leftEyeUpper2', 'rightEyebrowLower',
                  'leftEyebrowLower', 'rightEyeUpper1', 'leftEyeUpper0', 'noseBottom', 'midwayBetweenEyes',
                  'noseRightCorner', 'noseLeftCorner']

hyperparameters_RFC = {'n_estimators': 600, 'max_depth': 90, 'min_samples_split': 6, 'min_samples_leaf': 3,
                       'max_features': 'sqrt', 'bootstrap': False, 'criterion': 'entropy'}

hyperparameters_XGB = {'max_depth': 9,
                       'min_child_weight': 1,
                       'learning_rate': 0.2,
                       'subsample': 0.8,
                       'colsample_bytree': 1.0,
                       'gamma': 0,
                       'n_estimators': 600,
                       'use_label_encoder': False,
                       'eval_metric': 'rmse',
                       'objective': 'binary:logistic'}

hyperparameters_CB = {'bagging_temperature': 0.8607305832563434, 'bootstrap_type': 'MVS',
                      'colsample_bylevel': 0.917411003148779,
                      'depth': 8, 'grow_policy': 'SymmetricTree', 'iterations': 918, 'l2_leaf_reg': 8,
                      'learning_rate': 0.29287291117375575, 'max_bin': 231, 'min_data_in_leaf': 9, 'od_type': 'Iter',
                      'od_wait': 21, 'one_hot_max_size': 7, 'random_strength': 0.6963042728397884,
                      'scale_pos_weight': 1.924541179848884, 'subsample': 0.6480869299533999}

rf = RandomForestClassifier(**hyperparameters_RFC, random_state=150)
cb = CatBoostClassifier(**hyperparameters_CB)
xgb = XGBClassifier(**hyperparameters_XGB)
models = [
    ("XGBoost", xgb, "Blues"),
    ("Random Forest", rf, "Purples"),
    ("CatBoost", cb, "Reds"),
    ("Voting", VotingClassifier(estimators=[
        ("cb", cb),
        ('xgb', xgb),
        ('rf', rf),
    ], voting='hard'), "Greens")]

best_scores = {name: (0, 0) for name, model, color in models}

for i in range(1, 29):
    drop_columns = ['Filename', "is_stroke_face"]
    print(f"Top {i} regions")
    c = 0
    for region in ranked_regions:
        if c >= i:
            for index in FACE_INDEXES[region]:
                drop_columns.append(f"{region}_{index}")
        c += 1
    df = pd.read_csv('COMBO2.csv')
    target = df["is_stroke_face"]
    features = df.drop(drop_columns, axis=1)
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=150)
    for model in models:
        model[1].fit(X_train, y_train)
        y_pred = model[1].predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_scores[model[0]][0]:
            best_scores[model[0]] = (accuracy, i)
for name, (score, i) in best_scores.items():
    print(f"{name} model has the highest accuracy score of {score} with {i} regions")
