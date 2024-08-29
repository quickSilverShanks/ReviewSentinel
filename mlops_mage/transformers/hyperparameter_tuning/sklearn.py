import pickle

from mlops_mage.utils.basic_models.sklearn import load_class, tune_hyperparameters

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
# if 'test' not in globals():
#     from mage_ai.data_preparation.decorators import test


def read_pickle(filename):
    '''
    filename : filepath and filename(.pkl file) as a single string
    This function can be used to read any pickle file
    '''
    with open(filename, 'rb') as f:
        return pickle.load(f)


@transformer
def hyperparameters_tuning(data_splits, model_type, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data_splits: The output from the upstream GDP block
        model_type : The output from upstream custom block, contains model info
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here

    train_xy = read_pickle(data_splits['data_export'][0]['train_xy'])
    val_xy = read_pickle(data_splits['data_export'][0]['val_xy'])
    test_xy = read_pickle(data_splits['data_export'][0]['test_xy'])

    model_class = load_class(model_type)



    return data


# @test
# def test_output(output, *args) -> None:
#     """
#     Template code for testing the output of the block.
#     """
#     assert output is not None, 'The output is undefined'
