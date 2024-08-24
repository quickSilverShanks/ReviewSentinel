from sklearn.model_selection import StratifiedShuffleSplit


def data_splits(df_in):
    '''
    This function splits the data into train-val-test subsets in ratio of 0.800:0.175:0.025
    The statified splits have equal proportions of ratings ('rating' column must exist in 'df_in').
    '''
    splitter =StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=72)

    for train_index, test_index in splitter.split(df_in, df_in['rating']):
        df_train = df_in.iloc[train_index]
        df_test_tmp = df_in.iloc[test_index]

    val_splitter =StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=72)

    for val_index, test_index in val_splitter.split(df_test_tmp, df_test_tmp['rating']):
        df_val = df_test_tmp.iloc[val_index]
        df_test = df_test_tmp.iloc[test_index]
        
    return df_train, df_val, df_test