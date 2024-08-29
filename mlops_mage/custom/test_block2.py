if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom


@custom
def transform_custom(mtype_in, data_splits, *args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your custom logic here
    print(f"{type(mtype_in)}, {len(mtype_in)}, \n{mtype_in[0]}, \n{mtype_in[1]}")

    print(type(data_splits['data_export']), len(data_splits['data_export']))
    print(data_splits['data_export'][0])
    print(type(data_splits['data_export'][0]))
