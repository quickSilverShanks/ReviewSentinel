def get_xydata(df_in, cols_x, col_y):
    '''
    This function reads the input data and returns two ndarrays with dependent and independent features.
    It also returns the names of column(s) corresponding to the ndarrays.
    The input parameters 'cols_x' and 'col_y' should be list, irrespective of number of columns.
    Also, the col_y column should be integer target.
    Using global variable for target mapper on global target variable before passing the parameter to this function call is suggested.
    '''
    df_x = df_in[cols_x].copy()
    df_y = df_in[col_y].copy()

    data_x = df_x.values
    data_y = df_y.values.flatten()

    return data_x, data_y