from imblearn.under_sampling import RandomUnderSampler


def data_resampler(df_train, target):
    rsp = RandomUnderSampler()
    X_rsp, y_rsp = rsp.fit_resample(df_train, target["TARGET"])

    return X_rsp, y_rsp
