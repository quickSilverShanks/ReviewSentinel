import numpy as np
from sklearn import __all__
from sklearn.metrics import mean_squared_error
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from mlops_mage.utils.hyperparameters.shared import build_hyperparameters_space

HYPERPARAMETERS_WITH_CHOICE_INDEX = [
    'fit_intercept',
]


def load_class(module_and_class_name):
    """
    Loads a machine learning class from scikit-learn based on its module and class name.

    Args:
        module_and_class_name (str): The module and class name, separated by a dot (e.g., "linear_model.LogisticRegression").
            linear_model.LogisticRegression
            svm.SVC
            tree.DecisionTreeClassifier
            ensemble.RandomForestClassifier
            ensemble.GradientBoostingClassifier
            neural_network.MLPClassifier
            neighbors.KNeighborsClassifier
            naive_bayes.GaussianNB

    Returns:
        The loaded class sklearn BaseEstimator.        
    """
    parts = module_and_class_name.split('.')
    module_name = '.'.join(parts[:-1])
    class_name = parts[-1]

    # Check if the module is available in scikit-learn
    if module_name in __all__:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    else:
        raise ImportError(f"Module '{module_name}' not found in scikit-learn.")


def train_model(model, X_train, y_train, X_val = None, eval_metric = mean_squared_error, fit_params = None, y_val = None, **kwargs):
    model.fit(X_train, y_train, **(fit_params or {}))

    metrics = None
    y_pred = None
    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)

        rmse = eval_metric(y_val, y_pred, squared=False)
        mse = eval_metric(y_val, y_pred, squared=True)
        metrics = dict(mse=mse, rmse=rmse)

    return model, metrics, y_pred


def tune_hyperparameters(
    model_class, X_train, y_train, X_val, y_val, callback = None, eval_metric = mean_squared_error,
    fit_params = None, hyperparameters = None, max_evaluations = 50, random_state = 72):
    
    def __objective(
        params,
        X_train=X_train,
        X_val=X_val,
        callback=callback,
        eval_metric=eval_metric,
        fit_params=fit_params,
        model_class=model_class,
        y_train=y_train,
        y_val=y_val):

        model, metrics, predictions = train_model(
            model_class(**params),
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            eval_metric=eval_metric,
            fit_params=fit_params,
        )

        if callback:
            callback(
                hyperparameters=params,
                metrics=metrics,
                model=model,
                predictions=predictions,
            )

        return dict(loss=metrics['rmse'], status=STATUS_OK)

    space, choices = build_hyperparameters_space(
        model_class,
        random_state=random_state,
        **(hyperparameters or {}),
    )

    best_hyperparameters = fmin(
        fn=__objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evaluations,
        trials=Trials(),
    )

    # Convert choice index to choice value.
    for key in HYPERPARAMETERS_WITH_CHOICE_INDEX:
        if key in best_hyperparameters and key in choices:
            idx = int(best_hyperparameters[key])
            best_hyperparameters[key] = choices[key][idx]

    # fmin will return max_depth as a float for some reason
    for key in [
        'max_depth',
        'max_iter',
        'min_samples_leaf',
    ]:
        if key in best_hyperparameters:
            best_hyperparameters[key] = int(best_hyperparameters[key])

    return best_hyperparameters