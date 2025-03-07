import pandas as pd
from sklearn.model_selection import train_test_split
import os
import hashlib
import xgboost as xgb
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


def train_xgb(config):
    try:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        params = {
            "max_depth": int(config["max_depth"]) if config["max_depth"] is not None else None,
            "min_child_weight": config["min_child_weight"],
            "learning_rate": config["learning_rate"],
            "subsample": config["subsample"],
            "colsample_bytree": config["colsample_bytree"],
            "gamma": config["gamma"],
            "objective": config["objective"],
            "eval_metric": config["eval_metric"],
            "use_label_encoder": config["use_label_encoder"]
        }
        results = {}
        bst = xgb.train(params, dtrain, num_boost_round=int(config["n_estimators"]),
                        evals=[(dtest, "test")], evals_result=results, verbose_eval=False)
        accuracy = 1 - float(results['test']['rmse'][-1])  # Simplistic accuracy approximation
        return {"accuracy": accuracy}
    except Exception as e:
        logging.error(f"Error in train_xgb: {str(e)}")
        raise e


# Define search space for XGBoost
search_space = {
    "max_depth": tune.choice([None, 3, 4, 5, 6, 7, 8, 9, 10]),
    "min_child_weight": tune.choice([1, 2, 3, 4, 5]),
    "learning_rate": tune.choice([0.01, 0.02, 0.05, 0.1, 0.2]),
    "subsample": tune.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    "colsample_bytree": tune.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    "gamma": tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    "n_estimators": tune.choice([100, 200, 300, 400, 500]),
    "use_label_encoder": tune.choice([False]),
    "eval_metric": tune.choice(['rmse']),
    "objective": tune.choice(['binary:logistic'])
}

# Initialize Ray
ray.init(logging_level=logging.INFO)

# Run hyperparameter tuning
try:
    analysis = tune.run(
        train_xgb,
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
    best_params = best_config.copy()
    best_params.update({"n_estimators": int(best_config["n_estimators"])})
    best_xgb = xgb.XGBClassifier(**best_params)
    best_xgb.fit(X_train, y_train)
    final_accuracy = accuracy_score(y_test, best_xgb.predict(X_test))
    print(f"Final model accuracy: {final_accuracy}")

except Exception as e:
    print(f"An error occurred during Ray Tune execution: {str(e)}")

finally:
    ray.shutdown()
