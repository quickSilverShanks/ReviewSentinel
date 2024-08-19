'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Script Functionality

-- This script checks the results from the mlflow tracked experiments and selects the top 5 runs based on validation roc scores.
-- After that, it calculates the roc score of those models on the test set and saves the results to a new experiment.
-- Finally, it selects the model with the best roc score on the test set and registers it to the model registry.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score


def load_pickle(filename):
    '''
    filename : filepath and filename(.pkl file) as a single string
    This function can be used to read any pickle file
    '''
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    '''
    data_path : location where the processed data splits are saved
    params : parameters dictionary for the candidate model
    '''
    # read train and val split of the preprocessed data
    train_x, train_y, _, _ = load_pickle(
        os.path.join(data_path, "train.pkl")
        )
    val_x, val_y, _, _ = load_pickle(
        os.path.join(data_path, "val.pkl")
        )
    test_x, test_y, _, _ = load_pickle(
        os.path.join(data_path, "test.pkl")
        )
    
    # determine ml algorithm and setup parameter dict required for the model
    ml_algo = params['model']
    new_params = {k:int(v) if isinstance(v, str) and v.replace('.', '', 1).isdigit() and '.' not in v else
                    float(v) if isinstance(v, str) and v.replace('.', '', 1).isdigit() else v
                    for k, v in params.items() if k != 'model'}

    with mlflow.start_run():
        if ml_algo == 'Random Forest':
            model = RandomForestClassifier(**new_params)
        elif ml_algo == 'Logistic Regression':
            model = LogisticRegression(**new_params)
        elif ml_algo == 'SGD Classifier':
            model = SGDClassifier(**new_params)

        # fit model on train split
        model.fit(train_x, train_y)

        # Evaluate model on the validation and test splits
        val_yprobs = cross_val_predict(model, val_x, val_y, cv=3, method="predict_proba")[:,1]
        roc_score_val = roc_auc_score(val_y, val_yprobs)
        mlflow.log_metric("roc_score_val", roc_score_val)
        test_yprobs = cross_val_predict(model, test_x, test_y, cv=3, method="predict_proba")[:,1]
        roc_score_test = roc_auc_score(test_y, test_yprobs)
        mlflow.log_metric("roc_score_test", roc_score_test)


@click.command()
@click.option(
    "--data_path",
    default="./data/experiment_tracking",
    help="Location where the processed data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
@click.option(
    "--tracked_exps",
    default=("sklearn_baselines",),
    multiple=True,
    help="Tracked experiment(s) from which top_n models will be selected. Baseline models will always be considered, hence the default"
)
@click.option(
    "--candidate_exp",
    default="candidate_models",
    help="New experiment name in which top_n models will be evaluated and then best one will be registered to model registry"
)
def run_register_model(data_path, top_n, tracked_exps, candidate_exp):
    # combine default values of tracked_exps with provided values, if any
    default_tracked_exps = set(("sklearn_baselines",))
    tracked_exps = tuple(default_tracked_exps.union(set(tracked_exps)))

    # initialize mlflow; select a tracking uri : "sqlite:///mlflow.db" or "http://localhost:5000"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(candidate_exp)
    client = MlflowClient("sqlite:///mlflow.db")
    mlflow.sklearn.autolog()

    # initialize an empty list to store runs for candidate models
    all_runs = []

    # loop over each experiment and collect all runs
    for experiment_name in tracked_exps:
        experiment = client.get_experiment_by_name(experiment_name)
        runs = client.search_runs(
            experiment_ids=experiment.experiment_id,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=top_n,
            order_by=["metrics.roc_score DESC"]
        )
        all_runs.extend(runs)

    for run in all_runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # select the model with the highest test roc score
    experiment = client.get_experiment_by_name(candidate_exp)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.roc_score_test DESC"])[0]

    # Register the best model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="reviewsentinel_dev")



if __name__ == '__main__':
    run_register_model()