import pandas as pd
import lightgbm as lgb
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import json
from config.paths import *


def load_search_space(search_space_path):
    with open(search_space_path, "r") as f:
        raw_space = json.load(f)

    param_space = {}
    for param, spec in raw_space.items():
        if spec["type"] == "Real":
            param_space[param] = Real(
                spec["low"], spec["high"], prior=spec.get("prior", "uniform")
            )
        elif spec["type"] == "Integer":
            param_space[param] = Integer(spec["low"], spec["high"])
        else:
            raise ValueError(f"Unsupported type: {spec['type']}")

    return param_space


def get_best_hyperparameters(train_data, search_space_path):
    X_train = train_data.drop(columns="loan_status")
    y_train = train_data["loan_status"]

    lgb_model = lgb.LGBMClassifier(
        objective="binary", metric="binary_error", random_state=123
    )

    param_space = load_search_space(search_space_path)

    opt = BayesSearchCV(
        estimator=lgb_model,
        search_spaces=param_space,
        n_iter=5,
        cv=5,
        scoring="roc_auc",
        verbose=0,
    )

    opt.fit(X_train, y_train)
    return opt.best_params_


if __name__ == "__main__":
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    best_params = get_best_hyperparameters(train_data, "search_space.json")
    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
