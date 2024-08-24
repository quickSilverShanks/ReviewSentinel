import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf_vectorizer(df_in, text_col,  minperc, maxperc):
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
    # Use the passed vectorizer to transform the text column
    X = vectorizer.transform(df_in[text_col])

    # Convert the sparse matrix to a DataFrame
    col_names = ["f_" + wordcol for wordcol in vectorizer.get_feature_names_out()]
    tfidf_df = pd.DataFrame(X.toarray(), columns=col_names, index=df_in.index)

    # Merge the TF-IDF features with the original DataFrame using the index
    df_out = df_in.merge(tfidf_df, left_index=True, right_index=True)

    return df_out


def stdize_data(df_in, stdize_cols, scaler, fit=False):
    '''
    This function takes the input dataframe and the columns that need to be standardized.
    If fit is set to True it fits the provided StandardScaler on input data.
    Returns the data with standardized columns and StandardScaler object.
    '''
    if fit:
        scaled_data = scaler.fit_transform(df_in[stdize_cols])
    else:
        scaled_data = scaler.transform(df_in[stdize_cols])

    # convert the scaled ndarray data back into a DataFrame
    scaled_df = pd.DataFrame(scaled_data, columns=stdize_cols, index=df_in.index)

    # combine original unscaled features with scaled ones
    other_cols = [col for col in df_in.columns if col not in stdize_cols]
    dfscaled_out = pd.concat([df_in[other_cols], scaled_df], axis=1)

    return dfscaled_out, scaler


def get_features(df_train, df_val, df_test, minperc=0.005, maxperc=0.995):
    '''
    This function takes in train-val-test splits,
    trains a tf-idf vectorizer on train split and then applies it on all the 3 splits.
    Returns train-val-test splits with features created and the TfidfVectorizer and StandardScaler that was fit/used.
    '''
    # column with the cleaned reviews text
    text_col = 'cleaned_text'

    # fit the tf-idf vectorizer on train data
    vectorizer = get_tfidf_vectorizer(df_train, text_col, minperc, maxperc)

    # create tf-idf features in all the 3 splits
    df_train_wt_tfidf = get_tfidf_features(df_train, text_col, vectorizer)
    df_val_wt_tfidf = get_tfidf_features(df_val, text_col, vectorizer)
    df_test_wt_tfidf = get_tfidf_features(df_test, text_col, vectorizer)

    # standardize independent features
    stdize_cols = ["f_" + wordcol for wordcol in vectorizer.get_feature_names_out()] + ['rating']
    scaler = StandardScaler()
    df_train_out, scaler = stdize_data(df_train_wt_tfidf, stdize_cols, scaler, fit=True)
    df_val_out, _ = stdize_data(df_val_wt_tfidf, stdize_cols, scaler)
    df_test_out, _ = stdize_data(df_test_wt_tfidf, stdize_cols, scaler)

    return df_train_out, df_val_out, df_test_out, vectorizer, scaler