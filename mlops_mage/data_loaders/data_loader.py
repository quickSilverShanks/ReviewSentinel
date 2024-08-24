import requests
from io import BytesIO

import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
# if 'test' not in globals():
#     from mage_ai.data_preparation.decorators import test

@data_loader
def load_data(*args, **kwargs):
    """
    Loading data from github location.

    Returns:
        Dataframe with text reviews and other columns
    """

    response = requests.get(
        "https://raw.githubusercontent.com/quickSilverShanks/ReviewSentinel/main/data/fake%20reviews%20dataset.csv"
        )

    if response.status_code != 200:
        raise Exception(response.text)

    df = pd.read_csv(BytesIO(response.content))

    return df


# @test
# def test_output(output, *args) -> None:
#     """
#     Template code for testing the output of the block.
#     """
#     assert output is not None, 'The output is undefined'
