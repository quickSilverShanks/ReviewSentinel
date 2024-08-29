if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom


@custom
def transform_custom(d_in, *args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your custom logic here

    print(d_in[0][0].shape)

