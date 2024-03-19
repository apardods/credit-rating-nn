import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve

def make_fcnn_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def make_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(input_shape, 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=2, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def train_model(model, X_train, y_train, X_test, y_test, model_path):
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, restore_best_weights=True)
    history = model.fit(X_train,
                        y_train,
                        epochs=100,
                        batch_size=32,
                        validation_data=(X_test, y_test),
                        callbacks=[checkpoint, early_stopping],
                        verbose=2)
    return history, model
    
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

def find_optimal_threshold(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    max_index = np.argmax(f1_scores)
    optimal_threshold = thresholds[max_index]
    return optimal_threshold

def make_binary_classification(df, cols, target):
    X = df[cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.fit_transform(X_test)
    print('data ready')
    fcnn_model = make_fcnn_model(X_train.shape[1])
    cnn_model = make_cnn_model(X_train.shape[1])
    print('models compiled')

    if os.path.exists(r'..\models\fcnn_model.keras'):
        fcnn_model = load_model(r'..\models\fcnn_model.keras')
    else:
        history, fcnn_model = train_model(fcnn_model, X_train, y_train, X_test, y_test, r'..\models\fcnn_model.keras')
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