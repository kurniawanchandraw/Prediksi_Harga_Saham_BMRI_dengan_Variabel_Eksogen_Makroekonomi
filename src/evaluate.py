from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_forecast(y_true, y_pred):
    """
    Menghitung beberapa metrik evaluasi untuk hasil prediksi terhadap nilai sebenarnya.

    Metrik yang dihitung:
    - MSE (Mean Squared Error)
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - MAPE (Mean Absolute Percentage Error)
    - R2 (Coefficient of Determination)

    Args:
        y_true (array-like): Array atau list berisi nilai asli.
        y_pred (array-like): Array atau list berisi nilai prediksi.

    Returns:
        dict: Dictionary yang berisi nilai metrik evaluasi dengan keys
              'MSE', 'RMSE', 'MAE', 'MAPE', dan 'R2'.

    Raises:
        ValueError: Jika panjang y_true dan y_pred tidak sama.
        TypeError: Jika input bukan array-like numerik.
    """
    # Validasi input
    if len(y_true) != len(y_pred):
        raise ValueError("Panjang y_true dan y_pred harus sama.")
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Input harus berupa array 1 dimensi.")

    # Hitung metrik
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    # Hitung MAPE dengan penanganan pembagian nol
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.abs((y_true - y_pred) / y_true)
        mape = mape[~np.isinf(mape)]  # buang inf akibat pembagian nol
        mape = mape[~np.isnan(mape)]  # buang NaN jika ada
        mape = np.mean(mape) * 100 if len(mape) > 0 else np.nan

    r2 = r2_score(y_true, y_pred)

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}