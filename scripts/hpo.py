'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Script Functionality

-- This script reads preprocessed data's ndarrays and trains/validates randomforest models using hyperopt library
-- Uses mlflow to do experiment tracking with randomforest models
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import os
import pickle
import click
import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score


def load_pickle(filename):
    '''
    filename : filepath and filename(.pkl file) as a single string
    This function can be used to read any pickle file
    '''
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./data/experiment_tracking",
    help="Location where the processed data was saved"
)
@click.option(
    "--num_trials",
    default=15,
    help="The number of parameter evaluations for the optimizer to explore"
)
@click.option(
    "--exp_name",
    default="hyperopt_rfc",
    help="Name of mlflow experiment to keep track of trained models"
)
def run_optimization(data_path, num_trials, exp_name):
    # read train and val split of the preprocessed data
    train_x, train_y, _, _ = load_pickle(
        os.path.join(data_path, "train.pkl")
        )
    val_x, val_y, _, _ = load_pickle(
        os.path.join(data_path, "val.pkl")
        )

    # initialize mlflow; select a tracking uri : "sqlite:///mlflow.db" or "http://localhost:5000"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(exp_name)

    def objective(params):

        with mlflow.start_run():
            mlflow.set_tag("model", "hyperopt")
            mlflow.log_params(params)
            model = RandomForestClassifier(**params)
            model.fit(train_x, train_y)
            val_pred = model.predict(val_x)

            cv3_acc_train = cross_val_score(model, train_x, train_y, cv=3, scoring="accuracy")
            cv3_acc_val = cross_val_score(model, val_x, val_y, cv=3, scoring="accuracy")
            accuracy_val = accuracy_score(val_y, val_pred)
            val_yprobs = cross_val_predict(model, val_x, val_y, cv=3, method="predict_proba")[:,1]
            precision = precision_score(val_y, val_pred)
            recall = recall_score(val_y, val_pred)
            roc_score = roc_auc_score(val_y, val_yprobs)

            mlflow.log_param("model", "Random Forest")
            mlflow.log_metric("cv3_accuracy_train", sum(cv3_acc_train)/len(cv3_acc_train))
            mlflow.log_metric("cv3_accuracy_val", sum(cv3_acc_val)/len(cv3_acc_val))
            mlflow.log_metric("accuracy_val", accuracy_val)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("roc_score", roc_score)

        return {'loss': -1*roc_score, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 3, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 500, 1)),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 0.01, 0.05),
        'criterion' : hp.choice('criterion', ['gini', 'entropy']),
        'max_features' : hp.uniform('max_features', 0.1, 0.8),
        'random_state': 72
    }

    rstate = np.random.default_rng(72)  # for reproducible results
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )


if __name__ == '__main__':
    run_optimization()