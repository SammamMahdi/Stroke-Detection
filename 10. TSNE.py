import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

# Load data
df = pd.read_csv('COMBO2.csv')

drop_columns = ['Filename', "is_stroke_face"]

X = df.drop(drop_columns, axis=1)
y = df["is_stroke_face"]

# Hyperparameters
hyperparameters_RFC = {'n_estimators': 300, 'max_depth': 90, 'min_samples_split': 6, 'min_samples_leaf': 3,
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

# Define models
models = {
    # 'RandomForest': RandomForestClassifier(**hyperparameters_RFC, random_state=150),
    # 'XGBoost': XGBClassifier(**hyperparameters_XGB),
    'CatBoost': CatBoostClassifier(**hyperparameters_CB),
    'VotingClassifier': VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(**hyperparameters_RFC, random_state=150)),
        ('xgb', XGBClassifier(**hyperparameters_XGB)),
        ('cb', CatBoostClassifier(**hyperparameters_CB))
    ], voting='hard')
}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=150)

# Training and collecting feature representations
features_epochs = {model_name: [] for model_name in models}

for epoch in range(50, 550, 50):  # Simulate 10 epochs
    for name, model in models.items():
        model.fit(X_train, y_train)  # Train model
        # For deep learning, extract features from a specific layer
        # For sklearn, use the decision function or probabilities as features for TSNE
        if hasattr(model, 'decision_function'):
            features = model.decision_function(X_test)
        else:
            features = model.predict_proba(X_test)
        features_epochs[name].append(features)

        # Periodic evaluation (like per epoch)
        y_pred = model.predict(X_test)
        print(f"Epoch {epoch}, {name} Accuracy: {accuracy_score(y_test, y_pred)}")

# Applying T-SNE and plotting
for name, epochs_features in features_epochs.items():
    plt.figure(figsize=(15, 10))
    for i, features in enumerate(epochs_features):
        tsne = TSNE(n_components=2, random_state=42)
        X_embedded = tsne.fit_transform(features)
        plt.subplot(2, 5, i + 1)
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_test, cmap='viridis')
        plt.title(f'{name} Epoch {i + 1}')
    plt.show()
