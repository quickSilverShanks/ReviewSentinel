'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Script Functionality

-- This script reads preprocessed data's ndarrays and trains/validates several models.
-- Uses mlflow to do experiment tracking with sklearn models
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import os
import pickle
import click

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

import mlflow
import mlflow.sklearn


# list of baseline models to be trained and tracked
model_list = [
    (
        "Logistic Regression",
        {'solver':"lbfgs", 'max_iter':1000, 'random_state':72},
        LogisticRegression()
    ),
    (
        "SGD Classifier",
        {'max_iter':1000, 'tol':1e-3, 'loss':'modified_huber', 'random_state':72},
        SGDClassifier()
    ),
    (
        "Random Forest",
        {'n_estimators':200, 'min_samples_split':0.02, 'random_state':72},
        RandomForestClassifier()
    )
]


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
    "--exp_name",
    default="sklearn_baselines",
    help="Name of mlflow experiment to keep track of trained models"
)
@click.option(
    "--artifact_path",
    default="baseline_models",
    help="Location where model artifacts will be stored"
)
def run_train(data_path, exp_name, artifact_path):
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

    # train, evaluate and track models
    for model_name, mparams, model in model_list:
        with mlflow.start_run(run_name=model_name):
            model.set_params(**mparams)
            model.fit(train_x, train_y)
            val_pred = model.predict(val_x)

            mlflow.log_params(mparams)
            mlflow.sklearn.log_model(model, artifact_path)

            cv3_acc_train = cross_val_score(model, train_x, train_y, cv=3, scoring="accuracy")
            cv3_acc_val = cross_val_score(model, val_x, val_y, cv=3, scoring="accuracy")
            accuracy_val = accuracy_score(val_y, val_pred)
            val_yprobs = cross_val_predict(model, val_x, val_y, cv=3, method="predict_proba")[:,1]
            precision = precision_score(val_y, val_pred)
            recall = recall_score(val_y, val_pred)
            roc_score = roc_auc_score(val_y, val_yprobs)

            mlflow.log_param("model", model_name)
            mlflow.log_metric("cv3_accuracy_train", sum(cv3_acc_train)/len(cv3_acc_train))
            mlflow.log_metric("cv3_accuracy_val", sum(cv3_acc_val)/len(cv3_acc_val))
            mlflow.log_metric("accuracy_val", accuracy_val)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("roc_score", roc_score)



if __name__ == '__main__':
    run_train()