import pandas as pd

from mlops_mage.utils.data_preparation.text_cleaning import clean
from mlops.utils.data_preparation.splitters import data_splits
from mlops.utils.data_preparation.feature_engineering import get_features


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
# if 'test' not in globals():
#     from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    """
    Code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        DatFrames for train-val-test splits
        TfidfVectorizer and StandardScaler objects
    """
    # Transformation logic goes here

    review_col = kwargs.get('review_col')

    data_clean = clean(df_in, text_col=review_col)
    data_train, data_val, data_test = data_splits(data_clean)
    df_train, df_val, df_test, vectorizer, scaler = get_features(data_train, data_val, data_test)

    return df_train, df_val, df_test, vectorizer, scaler


# @test
# def test_output(output, *args) -> None:
#     """
#     Template code for testing the output of the block.
#     """
#     assert output is not None, 'The output is undefined'
