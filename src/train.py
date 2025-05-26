import tensorflow as tf

def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=100,
    batch_size=32,
    early_stopping_patience=15,
    reduce_lr_factor=0.5,
    reduce_lr_patience=5,
    callbacks=None,
    verbose=1
):
    """
    Melatih model TensorFlow Keras dengan callback EarlyStopping dan ReduceLROnPlateau.

    Args:
        model (tf.keras.Model): Model yang akan dilatih.
        X_train (array-like): Data input training.
        y_train (array-like): Target output training.
        X_val (array-like): Data input validasi.
        y_val (array-like): Target output validasi.
        epochs (int, optional): Jumlah epoch pelatihan. Default 100.
        batch_size (int, optional): Ukuran batch untuk pelatihan. Default 32.
        early_stopping_patience (int, optional): Jumlah epoch tanpa peningkatan sebelum menghentikan pelatihan lebih awal. Default 15.
        reduce_lr_factor (float, optional): Faktor pengurangan learning rate saat tidak ada peningkatan. Default 0.5.
        reduce_lr_patience (int, optional): Jumlah epoch tanpa peningkatan sebelum learning rate dikurangi. Default 5.
        callbacks (list, optional): Daftar callback tambahan yang ingin digunakan. Default None.
        verbose (int, optional): Verbosity mode pelatihan (0 = silent, 1 = progress bar, 2 = satu baris per epoch). Default 1.

    Returns:
        tf.keras.callbacks.History: Objek history yang berisi riwayat pelatihan.
    """
    # Setup default callbacks jika tidak ada input tambahan
    default_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=early_stopping_patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=reduce_lr_factor, patience=reduce_lr_patience
        ),
    ]

    if callbacks is not None:
        default_callbacks.extend(callbacks)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=default_callbacks,
        verbose=verbose,
    )
    return history
