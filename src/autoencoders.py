import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve

def build_autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    # This model maps an input to its reconstruction
    autoencoder = Model(input_layer, decoded)
    # This model maps an input to its encoded representation
    encoder = Model(input_layer, encoded)
    # This is a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder, encoder, decoder

def train_model(model, X_train, X_test):
    model.fit(X_train, X_train,
                    epochs=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(X_test, X_test))
    return model

def find_optimal_threshold(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    max_index = np.argmax(f1_scores)
    optimal_threshold = thresholds[max_index]
    return optimal_threshold

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    threshold = find_optimal_threshold(y_test, predictions)
    predictions_binary = (predictions > threshold).astype("int32")
    accuracy = accuracy_score(y_test, predictions_binary)
    precision = precision_score(y_test, predictions_binary)
    recall = recall_score(y_test, predictions_binary)
    f1 = f1_score(y_test, predictions_binary)
    auc = roc_auc_score(y_test, predictions)
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': auc
    }
    return metrics

def build_classifier(input_dim):
    input_layer = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(input_layer)
    x = Dense(32, activation='relu')(x)
    output_layer = Dense(1, activation='sigmoid')(x)  # Binary classification
    
    classifier = Model(inputs=input_layer, outputs=output_layer)
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return classifier

def classify_encoded(encoder, X_train, X_test, y_train, y_test, encoding_dim):
    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test)

    classifier_model = build_classifier(encoding_dim)
    classifier_model.fit(X_train_encoded, y_train, epochs=100, validation_split=0.2, verbose=1)
    metrics = evaluate_model(classifier_model, X_test_encoded, y_test)
    return metrics


def make_binary_classification(df, cols, target):
    X = df[cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mm = MinMaxScaler()
    X_train = mm.fit_transform(X_train)
    X_test = mm.fit_transform(X_test)
    print('data ready')
    input_dim = X_train.shape[1]  # Assuming X_train is your input features scaled
    encoding_dim = 32  # Example compression of input to a 32-dimensional representation

    autoencoder, encoder, _ = build_autoencoder(input_dim, encoding_dim)
    
    print('models compiled')

    if os.path.exists(r'..\models\autoencoder.keras'):
        autoencoder = load_model(r'..\models\autoencoder.keras')
    else:
        autoencoder = train_model(autoencoder, X_train, X_test)
    print('fcnn trained')

    if os.path.exists(r'..\models\cnn_model.keras'):
        cnn_model = load_model(r'..\models\cnn_model.keras')
    else:
        history, cnn_model = train_model(cnn_model, X_train, y_train, X_test, y_test, r'..\models\cnn_model.keras')
    print('cnn trained')

    fcnn_metrics = evaluate_model(fcnn_model, X_test, y_test)
    cnn_metrics = evaluate_model(cnn_model, X_test, y_test)
    print('evaluated')
    combined = {'FCNN': fcnn_metrics, 'CNN': cnn_metrics}
    return combined