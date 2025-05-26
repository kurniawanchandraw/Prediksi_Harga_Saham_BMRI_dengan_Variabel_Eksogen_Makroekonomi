import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Concatenate, LayerNormalization,
    MultiHeadAttention, Add, Conv1D, Activation, SpatialDropout1D
)
import numpy as np

def build_lstm_multi_input(seq_len, n_endog, n_exog, lstm_units=64, dropout_rate=0.2):
    """
    Bangun model LSTM dengan dua input (endogen dan eksogen).

    Args:
        seq_len (int): Panjang sequence input (timesteps).
        n_endog (int): Jumlah fitur endogen (target).
        n_exog (int): Jumlah fitur eksogen (fitur tambahan).
        lstm_units (int, optional): Jumlah neuron di layer LSTM. Default 64.
        dropout_rate (float, optional): Dropout rate untuk regularisasi. Default 0.2.

    Returns:
        tf.keras.Model: Model siap training.

    Raises:
        ValueError: Jika ada parameter input tidak valid (misal negatif atau nol).
        TypeError: Jika tipe parameter salah.
    """
    # Validasi input
    for name, val in zip(
        ['seq_len', 'n_endog', 'n_exog', 'lstm_units'], 
        [seq_len, n_endog, n_exog, lstm_units]
    ):
        if not isinstance(val, int) or val <= 0:
            raise ValueError(f"{name} harus integer positif, dapat: {val}")
    if not (0 <= dropout_rate <= 1):
        raise ValueError(f"dropout_rate harus di antara 0 dan 1, dapat: {dropout_rate}")

    input_endog = Input(shape=(seq_len, n_endog))
    input_exog = Input(shape=(seq_len, n_exog))
    x = Concatenate()([input_endog, input_exog])
    x = LSTM(lstm_units)(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(n_endog)(x)
    model = Model(inputs=[input_endog, input_exog], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_lstm_transformer(
    seq_len, n_endog, n_exog, lstm_units=64, transformer_heads=4,
    transformer_dim=32, dropout_rate=0.2
):
    """
    Model LSTM-Transformer untuk time series forecasting.

    Args:
        seq_len (int): Panjang sequence input (timesteps).
        n_endog (int): Jumlah fitur endogen (target).
        n_exog (int): Jumlah fitur eksogen (fitur tambahan).
        lstm_units (int, optional): Jumlah neuron di layer LSTM. Default 64.
        transformer_heads (int, optional): Jumlah head di MultiHeadAttention. Default 4.
        transformer_dim (int, optional): Dimensi key/query pada MultiHeadAttention. Default 32.
        dropout_rate (float, optional): Dropout rate untuk regularisasi. Default 0.2.

    Returns:
        tf.keras.Model: Model siap training.

    Raises:
        ValueError: Jika parameter input tidak valid.
    """
    # Validasi parameter
    for name, val in zip(
        ['seq_len', 'n_endog', 'n_exog', 'lstm_units', 'transformer_heads', 'transformer_dim'], 
        [seq_len, n_endog, n_exog, lstm_units, transformer_heads, transformer_dim]
    ):
        if not isinstance(val, int) or val <= 0:
            raise ValueError(f"{name} harus integer positif, dapat: {val}")
    if not (0 <= dropout_rate <= 1):
        raise ValueError(f"dropout_rate harus di antara 0 dan 1, dapat: {dropout_rate}")

    input_endog = Input(shape=(seq_len, n_endog))
    input_exog = Input(shape=(seq_len, n_exog))

    x_endog = LSTM(lstm_units)(input_endog)
    x_endog = Dropout(dropout_rate)(x_endog)

    exog_norm1 = LayerNormalization(epsilon=1e-6)(input_exog)
    attn_output = MultiHeadAttention(num_heads=transformer_heads, key_dim=transformer_dim)(exog_norm1, exog_norm1)
    attn_output = Dropout(dropout_rate)(attn_output)
    exog_res1 = Add()([input_exog, attn_output])
    exog_norm2 = LayerNormalization(epsilon=1e-6)(exog_res1)

    ff = Dense(transformer_dim * 4, activation='relu')(exog_norm2)
    ff = Dropout(dropout_rate)(ff)
    ff = Dense(n_exog)(ff)
    exog_out = Add()([exog_res1, ff])

    concat = Concatenate()([x_endog, tf.keras.layers.GlobalAveragePooling1D()(exog_out)])

    output = Dense(n_endog)(concat)

    model = Model(inputs=[input_endog, input_exog], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def tcn_block(x, filters, kernel_size=3, dilation_rate=1, dropout=0.2):
    """
    Satu blok Temporal Convolutional Network (TCN) dengan residual connection.

    Args:
        x (tf.Tensor): Input layer.
        filters (int): Jumlah filter Conv1D.
        kernel_size (int, optional): Ukuran kernel Conv1D. Default 3.
        dilation_rate (int, optional): Dilation rate Conv1D. Default 1.
        dropout (float, optional): Dropout rate. Default 0.2.

    Returns:
        tf.Tensor: Output layer setelah TCN block.

    Raises:
        ValueError: Jika parameter tidak valid.
    """
    if not isinstance(filters, int) or filters <= 0:
        raise ValueError(f"filters harus integer positif, dapat: {filters}")
    if not isinstance(kernel_size, int) or kernel_size <= 0:
        raise ValueError(f"kernel_size harus integer positif, dapat: {kernel_size}")
    if not isinstance(dilation_rate, int) or dilation_rate <= 0:
        raise ValueError(f"dilation_rate harus integer positif, dapat: {dilation_rate}")
    if not (0 <= dropout <= 1):
        raise ValueError(f"dropout harus di antara 0 dan 1, dapat: {dropout}")

    conv = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)(x)
    conv = SpatialDropout1D(dropout)(conv)
    conv = Activation('relu')(conv)
    res = Conv1D(filters, 1, padding='same')(x)
    out = Add()([res, conv])
    out = Activation('relu')(out)
    return out


def build_lstm_transformer_tcn(
    seq_len, n_endog, n_exog, lstm_units=32, filters=32,
    kernel_size=3, dilation_rates=None, transformer_heads=4,
    transformer_dim=32, dropout_rate=0.2
):
    """
    Model LSTM-Transformer-TCN untuk time series forecasting.

    Args:
        seq_len (int): Panjang sequence input (timesteps).
        n_endog (int): Jumlah fitur endogen.
        n_exog (int): Jumlah fitur eksogen.
        lstm_units (int, optional): Jumlah neuron LSTM. Default 32.
        filters (int, optional): Jumlah filter Conv1D di TCN. Default 32.
        kernel_size (int, optional): Ukuran kernel Conv1D. Default 3.
        dilation_rates (list or None, optional): Daftar dilation rate untuk blok TCN. Default [1, 2, 4, 8].
        transformer_heads (int, optional): Jumlah head MultiHeadAttention. Default 4.
        transformer_dim (int, optional): Dimensi key/query Transformer. Default 32.
        dropout_rate (float, optional): Dropout rate regularisasi. Default 0.2.

    Returns:
        tf.keras.Model: Model siap training.

    Raises:
        ValueError: Jika parameter tidak valid.
    """
    if dilation_rates is None:
        dilation_rates = [1, 2, 4, 8]

    # Validasi parameter
    int_params = {
        'seq_len': seq_len,
        'n_endog': n_endog,
        'n_exog': n_exog,
        'lstm_units': lstm_units,
        'filters': filters,
        'kernel_size': kernel_size,
        'transformer_heads': transformer_heads,
        'transformer_dim': transformer_dim,
    }
    for name, val in int_params.items():
        if not isinstance(val, int) or val <= 0:
            raise ValueError(f"{name} harus integer positif, dapat: {val}")

    if not (0 <= dropout_rate <= 1):
        raise ValueError(f"dropout_rate harus di antara 0 dan 1, dapat: {dropout_rate}")

    if not isinstance(dilation_rates, (list, tuple)) or not all(isinstance(d, int) and d > 0 for d in dilation_rates):
        raise ValueError("dilation_rates harus list/tuple berisi integer positif")

    input_endog = Input(shape=(seq_len, n_endog))
    input_exog = Input(shape=(seq_len, n_exog))

    # Transformer block untuk eksogen
    exog_norm1 = LayerNormalization(epsilon=1e-6)(input_exog)
    attn = MultiHeadAttention(num_heads=transformer_heads, key_dim=transformer_dim)(exog_norm1, exog_norm1)
    attn = Dropout(dropout_rate)(attn)
    exog_res1 = Add()([input_exog, attn])
    exog_norm2 = LayerNormalization(epsilon=1e-6)(exog_res1)

    ff = Dense(transformer_dim * 4, activation='relu')(exog_norm2)
    ff = Dropout(dropout_rate)(ff)
    ff = Dense(n_exog)(ff)
    exog_out = Add()([exog_res1, ff])

    # TCN blocks untuk endogen
    x = input_endog
    for d in dilation_rates:
        x = tcn_block(x, filters=filters, kernel_size=kernel_size, dilation_rate=d, dropout=dropout_rate)
    x = Dropout(dropout_rate)(x)

    # Gabungkan fitur TCN dan Transformer eksogen
    concat = Concatenate()([x, exog_out])

    # LSTM setelah concat
    lstm_out = LSTM(lstm_units)(concat)
    lstm_out = Dropout(dropout_rate)(lstm_out)

    output = Dense(n_endog)(lstm_out)

    model = Model(inputs=[input_endog, input_exog], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model