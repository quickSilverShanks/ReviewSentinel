'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Script Functionality

-- This script has preprocessing functions to create tf-idf features. Train-Val-Test split should already exist to use this script.
-- As it is currently, the input pickle file should have the cleaned text column available already(it was created in the data_profiling notebook).
-- There is already a function to create new tf-idf vectorizer in this script. Instead, user can also provide a path for available vectorizer to be used.
-- Ultimately, it takes preprocessed data as input and converts it into usable ndarrays and stores it as pickle file.
   This stored pickle file can be read to get 2 ndarrays with train features/targets and 2 lists of corresponding arrays' column names.

Additional functionality to clean the review text and standardize data will be added in this script in future modifications.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import os
import pickle
import click
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def read_pickledata(filename):
    '''
    filename : filepath and filename(data pickle) as a single string
    This function will exclusively be used to import dataframe
    '''
    input_df = pd.read_pickle(filename)
    return input_df


def read_pickle(filename):
    '''
    filename : filepath and filename(.pkl file) as a single string
    This function can be used to read any pickle file
    '''
    with open(filename, 'rb') as f:
        return pickle.load(f)


def dump_pickle(obj, filename):
    '''
    obj : the object that needs to be dumped as pickle
    filename : string path and name of the destination pickle file
    '''
    with open(filename, 'wb') as f_out:
        pickle.dump(obj, f_out)


def get_tfidf_vectorizer(df_in, text_col, fileout_vect, fit_vect, minperc=0.005, maxperc=0.995):
    '''
    Fits a TF-IDF vectorizer if it does not exist(fit_vect set to True), or else just loads it.
    Saves the fitted vectorizer as a pickle file output(fileout_vect).
    '''
    if fit_vect:
        vectorizer = fit_tfidf_vectorizer(df_in, text_col,  minperc, maxperc)
        # create folder for fileout_vect unless it already exists
        folder_location = os.path.dirname(fileout_vect)
        os.makedirs(folder_location, exist_ok=True)
        dump_pickle(vectorizer, fileout_vect)
    else:
        vectorizer = read_pickle(fileout_vect)
    
    return vectorizer


def fit_tfidf_vectorizer(df_in, text_col, minperc, maxperc):
    '''
    Fits a TF-IDF vectorizer on the specified text column and returns the vectorizer.
    Considers words with occurence percentage in the range [minperc, maxperc]
    '''
    # tf-idf vectorization
    vectorizer = TfidfVectorizer(min_df=minperc, max_df=maxperc)
    vectorizer.fit(df_in[text_col])

    return vectorizer


def get_tfidf_features(df_in, text_col, vectorizer):
    '''
    Returns dataframe with tf-idf features created from the 'text_col'
    using the provided vectorizer.
    '''
    # use the passed vectorizer to transform the text column
    X = vectorizer.transform(df_in[text_col])

    # convert the sparse matrix to a DataFrame
    col_names = ["f_" + wordcol for wordcol in vectorizer.get_feature_names_out()]
    tfidf_df = pd.DataFrame(X.toarray(), columns=col_names, index=df_in.index)

    # merge the TF-IDF features with the original DataFrame using the index
    df_out = df_in.merge(tfidf_df, left_index=True, right_index=True)

    return df_out


def get_xydata(dframe, vectorizer, cols_x, col_y, col_y_mapper):

    '''
    Input Parameters:
     Reads the file 'file_name' from location 'file_path' which has the column 'col_text'.
     Numerical features will be created on this text column which will be required for model training/evaluation.
     dframe : dataframe with original columns and the created tf-idf features
     vectorizer : tf-idf vectorizer object
     cols_x : list of independent features from the file 'file_name' required in 'df_x'
     col_y : list with a single target column. Its mandatory to have 'CG' in this column for computer generated reviews
     col_y_mapper : dictionary to map text contents of 'col_y' to integers
    Outputs:
     'df_x' and 'df_y': numpy nd-arrays which contain the independent features and target variable respectively
     'cols_x' and 'cols_y' : lists containing column names corresponding to the columns of output matrices
     
    '''
    assert all([col in dframe.columns for col in cols_x + col_y]), "required column(s) from input parameters not found"

    # Load the feature names from vectorizer object
    tfidf_features = ["f_" + wordcol for wordcol in vectorizer.get_feature_names_out()]

    df_x = dframe[tfidf_features + cols_x].copy()
    df_y = dframe[col_y].copy()
    df_y[col_y[0]] = df_y[col_y[0]].map(col_y_mapper)

    return df_x.values, df_y.values.flatten(), list(df_x.columns), list(df_y.columns)


def preprocess(df_in, target_col, vectorizer):
    '''
    df_in : dataframe that needs to be preprocessed
    target_col : target column, should be cleaned text
    vectorizer : tf-idf vectorizer object
    Outputs:
     'df_x' and 'df_y': numpy nd-arrays which contain the independent features and target variable respectively.
     'cols_x' and 'cols_y' : lists containing column names corresponding to the columns of output matrices.
    '''
    dframe_tfidf = get_tfidf_features(df_in, target_col, vectorizer)
    cols_x = ['rating']
    col_y = ['label']
    col_y_mapper = {'CG':1, 'OR':0}
    data_x, data_y, data_xcols, data_ycols = get_xydata(dframe_tfidf, vectorizer, cols_x, col_y, col_y_mapper)
    return data_x, data_y, data_xcols, data_ycols


@click.command()
@click.option(
    "--inp_data_path",
    help="Location of input pickle files with cleaned text column(named as 'cleaned_text')"
)
@click.option(
    "--dest_path",
    help="Location where the resulting files will be saved"
)
@click.option(
    "--datasets",
    default=("processed_reviews_train_v1.pkl", "processed_reviews_val_v1.pkl", "processed_reviews_test_v1.pkl"),
    help="Names of pickled train, val and test dataframes with cleaned_text column",
    multiple=True
)
@click.option(
    "--fileout_vect",
    help="Location and name of tf-idf vectorizer pickle file"
)
@click.option(
    "--fit_vect",
    is_flag=True,
    help="Fit tf-idf vectorizer flag. True, if tf-idf vectorizer needs to be fitted; False, if the vectortizer already exists"
)
def run_data_prep(inp_data_path, dest_path, datasets, fileout_vect, fit_vect):
    # load pickle files
    print("INFO : loading pickle files")
    df_train = read_pickledata(
        os.path.join(inp_data_path, f"{datasets[0]}")
        )
    df_val = read_pickledata(
        os.path.join(inp_data_path, f"{datasets[1]}")
        )
    df_test = read_pickledata(
        os.path.join(inp_data_path, f"{datasets[2]}")
        )
    
    # target column that has the cleaned reviews text
    target = 'cleaned_text'

    # load tf-idf vectorizer
    print("INFO : loading tf-idf vectorizer")
    vectorizer = get_tfidf_vectorizer(df_train, target, fileout_vect, fit_vect)

    print("INFO : preprocessing data splits")
    train = preprocess(df_train, target, vectorizer)
    val = preprocess(df_val, target, vectorizer)
    test = preprocess(df_test, target, vectorizer)

    # create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # save preprocessed datasets
    print("INFO : saving preprocessed pickle files")
    dump_pickle(train, os.path.join(dest_path, "train.pkl"))
    dump_pickle(val, os.path.join(dest_path, "val.pkl"))
    dump_pickle(test, os.path.join(dest_path, "test.pkl"))



if __name__=="__main__":
    run_data_prep()