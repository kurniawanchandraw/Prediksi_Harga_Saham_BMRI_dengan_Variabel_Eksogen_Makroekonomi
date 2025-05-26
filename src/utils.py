import numpy as np
from sklearn.preprocessing import RobustScaler

def create_sequences(data, seq_len, endog_cols, exog_cols, lags):
    """
    Membuat data sequence untuk input model time series.

    Args:
        data (pd.DataFrame): Dataframe berisi fitur.
        seq_len (int): Panjang sequence input.
        endog_cols (list): List nama kolom endogen (target).
        exog_cols (list): List nama kolom eksogen (fitur tambahan).
        lags (int): Jumlah lag fitur eksogen.

    Returns:
        tuple: (X_endog, X_exog, y) sebagai numpy array.
    """
    X_endog, X_exog, y = [], [], []
    exog_lag_cols = [f'{col}_lag{lag}' for col in exog_cols for lag in range(1, lags+1)]
    for i in range(seq_len, len(data)):
        X_endog.append(data[endog_cols].iloc[i-seq_len:i].values)
        X_exog.append(data[exog_lag_cols].iloc[i-seq_len:i].values)
        y.append(data[endog_cols].iloc[i].values)
    return np.array(X_endog), np.array(X_exog), np.array(y)


def inverse_transform(scaler, y_scaled, X_exog_sample, exog_cols, lags):
    """
    Melakukan inverse transform hasil scaling untuk output dan eksogen.

    Args:
        scaler (RobustScaler): Scaler yang sudah fit.
        y_scaled (np.array): Data target hasil prediksi atau asli, sudah diskalakan.
        X_exog_sample (np.array): Sample data eksogen terakhir sesuai urutan lag.
        exog_cols (list): List nama fitur eksogen.
        lags (int): Jumlah lag fitur eksogen.

    Returns:
        np.array: Data asli hasil inverse transform.
    """
    exog_lag_cols = [f'{col}_lag{lag}' for col in exog_cols for lag in range(1, lags+1)]
    dummy = np.hstack([y_scaled, X_exog_sample[:, -1, :]])
    inv = scaler.inverse_transform(dummy)
    return inv[:, 0]


def multi_step_forecast(model, X_endog_last, X_exog_last, steps, lags):
    """
    Forecast multi-step ke depan secara iteratif,
    menggunakan prediksi sebelumnya sebagai input berikutnya,
    dan menggunakan nilai eksogen terakhir untuk eksogen masa depan.

    Args:
        model : trained keras model
        X_endog_last : np.array shape (seq_len, n_endog), sequence terakhir endogen
        X_exog_last : np.array shape (seq_len, n_exog), sequence terakhir eksogen
        steps : int, jumlah langkah ke depan yang di-forecast
        lags : int, jumlah lag eksogen

    Returns:
        np.array shape (steps, n_endog), hasil forecast
    """
    preds = []
    endog_seq = X_endog_last.copy()
    exog_seq = X_exog_last.copy()

    for _ in range(steps):
        pred = model.predict([endog_seq[np.newaxis, :, :], exog_seq[np.newaxis, :, :]])[0]
        preds.append(pred)

        # Geser sequence endogen dan tambahkan prediksi terbaru
        endog_seq = np.vstack([endog_seq[1:], pred[np.newaxis, :]])

        # Geser sequence eksogen, isi dengan nilai terakhir (asumsi konstan)
        exog_seq = np.vstack([exog_seq[1:], exog_seq[-1:]])

    return np.array(preds)
