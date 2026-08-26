import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def create_lstm_sequences(df, features, target, sequence_length=24):
    """
    Constructs sequence windows city-by-city to prevent target leakage
    and ensure chronological ordering.
    
    Returns:
        X_seq: np.ndarray of shape (N, sequence_length, len(features))
        y_seq: np.ndarray of shape (N,)
    """
    X_seq_list = []
    y_seq_list = []
    
    for city, group in df.groupby("city"):
        # Ensure chronological sorting
        sorted_group = group.sort_values("timestamp").reset_index(drop=True)
        X_data = sorted_group[features].values
        y_data = sorted_group[target].values
        
        for i in range(sequence_length, len(sorted_group)):
            X_seq_list.append(X_data[i - sequence_length:i])
            y_seq_list.append(y_data[i])
            
    return np.array(X_seq_list, dtype=np.float32), np.array(y_seq_list, dtype=np.float32)

def build_lstm_model(input_shape):
    """
    Builds and compiles a lightweight LSTM neural network model for forecasting.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["mae"]
    )
    return model

def train_lstm(model, X_train, y_train, epochs=30, batch_size=32, validation_data=None):
    """
    Trains the LSTM model with early stopping.
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss" if validation_data else "loss",
            patience=5,
            restore_best_weights=True
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1
    )
    return history
