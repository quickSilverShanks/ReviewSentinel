# Use this block to test run small code features/blocks and imports to check if it works

import os
import ast
import json
import mlflow
import nltk

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom


@custom
def transform_custom(data_in, *args, **kwargs):
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

    print(f"Input from previous block : {type(data_in)}, {len(data_in)}")
