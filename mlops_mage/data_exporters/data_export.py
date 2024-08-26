import ast
import pickle

from mage_ai.io.file import FileIO
from mlops_mage.utils.data_preparation.feature_selector import get_xydata

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data_to_file(data, **kwargs) -> None:
    """
    Template for exporting data to filesystem.

    Docs: https://docs.mage.ai/design/data-loading#fileio

    Args:
        data : tuple with train-val-test splits of preprocessed i/p data
               and TfidfVectorizer and StandardScaler objects
    
    Returns:
        None, exports 5 pickle files that contain ndarrays for train-val-test splits (as tuples of and y features)
               and TfidfVectorizer and StandardScaler objects
    """
    df_train, df_val, df_test, vectorizer, scaler = data

    target_col = kwargs.get('target')
    target_mapper = ast.literal_eval(kwargs.get('target_mapper'))

    df_train[target_col] = df_train[target_col].map(target_mapper)
    df_val[target_col] = df_val[target_col].map(target_mapper)
    df_test[target_col] = df_test[target_col].map(target_mapper)

    cols_x = ["f_" + wordcol for wordcol in vectorizer.get_feature_names_out()] + ['rating']
    col_y = [target_col]

    train_x, train_y = get_xydata(df_train, cols_x, col_y)
    val_x, val_y = get_xydata(df_val, cols_x, col_y)
    test_x, test_y = get_xydata(df_test, cols_x, col_y)

    filepath = './artifacts/workflow_orchestration/'
    with open(filepath+'train.pickle', 'wb') as f_out:
        pickle.dump((train_x, train_y), f_out)
    with open(filepath+'val.pickle', 'wb') as f_out:
        pickle.dump((val_x, val_y), f_out)
    with open(filepath+'test.pickle', 'wb') as f_out:
        pickle.dump((test_x, test_y), f_out)
    with open(filepath+'vectorizer.pickle', 'wb') as f_out:
        pickle.dump(vectorizer, f_out)
    with open(filepath+'scaler.pickle', 'wb') as f_out:
        pickle.dump(scaler, f_out)
    
    # return (train_x, train_y), (val_x, val_y), (test_x, test_y), vectorizer, scaler
