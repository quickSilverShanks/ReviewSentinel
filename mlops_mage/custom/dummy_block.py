# Use this block to test run small code features/blocks and imports to check if it works

import os
import ast
import json
import mlflow
import nltk

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def transform_custom(*args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your custom logic here
    
    print("Block Execution Successful!!")
    print(os.getcwd())   # /home/src
    print(os.listdir('./artifacts'))
    target_mapper = kwargs.get('target_mapper')
    print(type(target_mapper), target_mapper)   # <class 'str'> {'CG':1, 'OR':0}
    target_mapper_dict = ast.literal_eval(kwargs.get('target_mapper'))
    print(type(target_mapper_dict), target_mapper_dict)
    # return {}


# @test
# def test_output(output, *args) -> None:
#     """
#     Template code for testing the output of the block.
#     """
#     assert output is not None, 'The output is undefined'
