import pandas as pd
from sklearn.model_selection import train_test_split
import os
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import logging

# Load and prepare data
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


def train_rf(config):
    try:
        rf = RandomForestClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]) if config["max_depth"] is not None else None,
            min_samples_split=int(config["min_samples_split"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            max_features=config["max_features"],
            bootstrap=config["bootstrap"],
            criterion=config["criterion"],
            random_state=150
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return {"accuracy": accuracy}
    except Exception as e:
        logging.error(f"Error in train_rf: {str(e)}")
        raise e


# Define expanded search space for RandomForest
search_space = {
    "n_estimators": tune.choice([100, 200, 300, 400, 500, 600, 700, 800]),
    "max_depth": tune.choice([None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
    "min_samples_split": tune.choice([2, 4, 6, 8, 10, 12, 14, 16]),
    "min_samples_leaf": tune.choice([1, 2, 3, 4, 5, 6, 7, 8]),
    "max_features": tune.choice(['sqrt', 'log2', 0.25, 0.5, 0.75]),
    "bootstrap": tune.choice([True, False]),
    "criterion": tune.choice(['gini', 'entropy'])
}

# Initialize Ray
ray.init(logging_level=logging.INFO)

# Run hyperparameter tuning
try:
    analysis = tune.run(
        train_rf,
        config=search_space,
        num_samples=200,
        scheduler=ASHAScheduler(metric="accuracy", mode="max", max_t=1000),
        search_alg=OptunaSearch(metric="accuracy", mode="max"),
        resources_per_trial={"cpu": 4, "gpu": 0},
        verbose=2,
        trial_dirname_creator=lambda trial: hashlib.md5(trial.trial_id.encode()).hexdigest()[:10],
        storage_path=os.path.expanduser("~/ray_results"),
        raise_on_failed_trial=False
    )
    print("Ray Tune completed.")

    best_config = analysis.get_best_config(metric="accuracy", mode="max")
    print("Best hyperparameters found:", best_config)

    # Train the final model with the best hyperparameters
    best_rf = RandomForestClassifier(**best_config, random_state=150)
    best_rf.fit(X_train, y_train)
    final_accuracy = accuracy_score(y_test, best_rf.predict(X_test))
    print(f"Final model accuracy: {final_accuracy}")

except Exception as e:
    print(f"An error occurred during Ray Tune execution: {str(e)}")

finally:
    ray.shutdown()
