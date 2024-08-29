if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
# if 'test' not in globals():
#     from mage_ai.data_preparation.decorators import test


@custom
def sklearn_models(*args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    skl_models : comma separated strings
        linear_model.LogisticRegression
        svm.SVC
        tree.DecisionTreeClassifier
        ensemble.RandomForestClassifier
        ensemble.GradientBoostingClassifier
        neural_network.MLPClassifier
        neighbors.KNeighborsClassifier
        naive_bayes.GaussianNB

    Returns:
        Tuple of sklearn models and their metadata dictionary
    """
    
    model_names = kwargs.get(
        'skl_models', 'linear_model.LogisticRegression, tree.DecisionTreeClassifier'
        )
    
    child_data = [
        model_name.strip() for model_name in model_names.split(',')
        ]
    
    child_metadata = [
        dict(block_uuid=model_name.split('.')[-1]) for model_name in child_data
        ]

    return child_data, child_metadata


# @test
# def test_output(output, *args) -> None:
#     """
#     Template code for testing the output of the block.
#     """
#     assert output is not None, 'The output is undefined'
