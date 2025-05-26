import matplotlib.pyplot as plt

def plot_train_val_forecast(y_train_true, y_train_pred, y_test_true, y_test_pred, 
                            forecast_values, forecast_steps):
    """
    Visualisasi gabungan hasil train, test dan forecast multistep.

    Args:
        y_train_true (array-like): Data target asli train.
        y_train_pred (array-like): Data prediksi train.
        y_test_true (array-like): Data target asli test.
        y_test_pred (array-like): Data prediksi test.
        forecast_values (array-like): Hasil forecast multistep.
        forecast_steps (int): Jumlah langkah forecast ke depan.
    """
    plt.figure(figsize=(14,7))

    plt.plot(range(len(y_train_true)), y_train_true, label='Train Actual', color='black', linewidth=0.6)
    plt.plot(range(len(y_train_pred)), y_train_pred, label='Train Predicted', color='orange', linewidth=0.6)

    start_test = len(y_train_true)
    plt.plot(range(start_test, start_test + len(y_test_true)), y_test_true, label='Test Actual', color='blue', linewidth=0.6)
    plt.plot(range(start_test, start_test + len(y_test_pred)), y_test_pred, label='Test Predicted', color='purple', linewidth=0.6)

    start_forecast = start_test + len(y_test_true)
    plt.plot(range(start_forecast, start_forecast + forecast_steps), forecast_values, 
             label=f'{forecast_steps}-Step Forecast', color='red', linestyle='--', linewidth=1)

    plt.title('Train & Test Actual vs Predicted + Multi-Step Forecast')
    plt.xlabel('Time Step')
    plt.ylabel('Harga Saham BMRI')
    plt.legend()
    plt.grid(True)
    plt.show()
