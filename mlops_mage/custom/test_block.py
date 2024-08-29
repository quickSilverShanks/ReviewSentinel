import pandas as pd

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom


@custom
def transform_custom(data, *args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your custom logic here
    print(type(data), data.keys())
    data_content = data['data_export']
    print(type(data_content), len(data_content))
    list_item1 = data_content[0]
    print(type(list_item1), len(list_item1))

    train_x, train_y = data['data_export'][0]
    
    dx = pd.DataFrame(train_x)
    dx.to_csv('./artifacts/tmp/dx_train.csv')

    # return {}
