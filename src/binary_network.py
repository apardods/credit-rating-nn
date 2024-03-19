import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from tensorflow.keras.metrics import Precision, Recall, AUC, F1Score, Accuracy

def build_fcnn_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.P])
    return model

def build_lstm_model(input_shape, output_units=1, output_activation='sigmoid'):  # Adjust the output layer based on the task
    model = Sequential([
        Embedding(input_dim=10000, output_dim=64, input_length=input_shape[1]),  # Adjust 'input_dim' and 'output_dim' based on your data
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(output_units, activation=output_activation)
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # Adjust loss function based on the task
                  metrics=['accuracy'])  # Adjust metrics based on the task
    return model