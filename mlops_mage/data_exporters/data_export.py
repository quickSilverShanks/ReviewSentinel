from mage_ai.io.file import FileIO
from mlops.utils.data_preparation.feature_selector import get_xydata

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data_to_file(data, **kwargs) -> None:
    """
    Template for exporting data to filesystem.

    Docs: https://docs.mage.ai/design/data-loading#fileio
    """
    df_train, df_val, df_test, vectorizer, _ = data

    target_col = kwargs.get('target')
    target_mapper = kwargs.get('target_mapper')

    df_train[target_col] = df_train[target_col].map(target_mapper)
    df_val[target_col] = df_val[target_col].map(target_mapper)
    df_test[target_col] = df_test[target_col].map(target_mapper)

    cols_x = ["f_" + wordcol for wordcol in vectorizer.get_feature_names_out()] + ['rating']
    col_y = [target_col]

    train_x, train_y = get_xydata(df_train, cols_x, col_y)
    val_x, val_y = get_xydata(df_val, cols_x, col_y)
    test_x, test_y = get_xydata(df_test, cols_x, col_y)

    filepath = '/mnt/d/GitHub/ReviewSentinel/artifacts/workflow_orchestration/'
    FileIO().export((train_x, train_y), filepath+'train.pickle')
    FileIO().export((val_x, val_y), filepath+'val.pickle')
    FileIO().export((test_x, test_y), filepath+'test.pickle')
