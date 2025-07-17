import os
import warnings
import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2
# from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from xgboost import XGBClassifier
import xgboost as xgb



os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding,Conv2D, MaxPooling2D, Flatten 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from typing import TYPE_CHECKING, Counter
 
if TYPE_CHECKING:
    from keras import models
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Embedding, Conv2D, MaxPooling2D,Flatten
    from keras.callbacks import EarlyStopping
    from keras.optimizers import Adam
    from keras.preprocessing.sequence import pad_sequences
    from keras.src.legacy.preprocessing.text import Tokenizer

from image_utils import Path, readTrafficSigns
from preprocessing import train_edited_images,test_edited_images, labels_test,labels_train

def get_classification_report(y_true, y_pred, dataset_name="Test"):
    """
    Simple classification report function
    """
    print(f"\n=== {dataset_name} Set Classification Report ===")
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred))
    
    return accuracy




cattegories = list(range(0,43))

# numeric_labels = [int(value) for value in cleaned_labels]
X = np.array(train_edited_images)
y = np.array([int(label) for label in labels_train])

print(y.dtype)
# print()

# unique_classes = len(np.unique(y))
# total_samples = y.shape[0]
# print(unique_classes, total_samples)

# class_counts = Counter(y)
# print("Class distribution:")
# for class_id in sorted(class_counts.keys()):
#     print(f"Class {class_id}: {class_counts[class_id]} samples")

# print(f"\nMin samples per class: {min(class_counts.values())}")
# print(f"Max samples per class: {max(class_counts.values())}")

# if total_samples > 20 and unique_classes > (0.5 * total_samples):
#     warnings.warn(
#         f"The number of unique classes ({unique_classes}) is greater than 50% "
#         f"of the number of samples ({total_samples}).",
#         UserWarning,
#         stacklevel=2,
#     )

"""Tikrina ar y turi str elementu"""
# if np.issubdtype(y.dtype, np.str_) or np.issubdtype(y.dtype, np.object_):
#     print("The array contains strings.")
# else:
#     print("The array does not contain strings.")

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)



# normalized the training and validation sets by dividing by 255
X_train_normalized = X_train / 255.0  # normalized training images
X_val_normalized = X_val / 255.0  # normalized validation images

print(X_train_normalized.shape, X_val_normalized.shape)

# print(f"X_train_normalized shape: {X_train_normalized.shape}, dtype: {X_train_normalized.dtype}")
# print(f"X_val_normalized shape: {X_val_normalized.shape}, dtype: {X_val_normalized.dtype}")

X_train_normalized_flat = X_train_normalized.reshape(X_train_normalized.shape[0], -1)
X_val_normalized_flat = X_val_normalized.reshape(X_val_normalized.shape[0], -1)
"""Pasiklausti"""
# smote = SMOTE(random_state=42, k_neighbors=3)
# X_train_smote, y_train_smote = smote.fit_resample(X_train_normalized_flat, y_train)

# 4. Train on balanced data

# Your model architecture (same as yours)
model = Sequential()

model.add(Conv2D(32, (2,2), activation='relu', input_shape=(96,96,1)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (2,2), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(96, (2,2), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten(name='flatten'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))

model.add(Dropout(0.3))
model.add(Dense(43, activation='softmax'))

model.compile(
    optimizer=Adam(learning_rate=0.002),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

print(">>> Model Architecture")
model.summary()

print(">>> Training CNN")
history_cnn = model.fit(
    X_train_normalized, y_train,
    validation_data=(X_val_normalized, y_val),
    batch_size=128,
    epochs=50,
    callbacks=[early_stop],
    verbose=2
)

# Save model
model.save("cnn_model.keras")

# PROPER EVALUATION FUNCTIONS
def evaluate_model(model, X_test, y_test, dataset_name="Test"):
    """Simple model evaluation function"""
    print(f"\n=== {dataset_name} Set Evaluation ===")
    
    # Keras evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Manual prediction
    predictions = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    sklearn_accuracy = accuracy_score(y_test, pred_classes)
    
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Sklearn Accuracy: {sklearn_accuracy:.4f} ({sklearn_accuracy*100:.2f}%)")
    
    return test_accuracy, pred_classes, predictions

def get_classification_report(y_true, y_pred, dataset_name="Test"):
    """Simple classification report function"""
    print(f"\n=== {dataset_name} Set Classification Report ===")
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred))
    
    return accuracy

def plot_training_history(history):
    """Plot training history graphs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# CORRECTED EVALUATION APPROACH

# 1. Prepare test data properly
X_test = np.array(test_edited_images)
X_test_normalizedd = X_test / 255.0
Y_test= np.array([int(label) for label in labels_test])

# ⚠️ IMPORTANT: You need y_test (true labels for test set)
# If you don't have y_test, you can't calculate accuracy!
# Make sure you have: y_test = np.array(test_labels)

# 2. Evaluate on your existing validation set
print("="*50)
print("VALIDATION SET EVALUATION")
print("="*50)

# normalized your validation data (if not already done)


val_acc, val_pred, val_proba = evaluate_model(model, X_val_normalized, y_val, "Validation")
val_report_acc = get_classification_report(y_val, val_pred, "Validation")

# 3. Test set predictions (no labels available yet)
print("="*50)
print("TEST SET PREDICTIONS")
print("="*50)

print("Generating predictions for test set...")
test_predictions_prob = model.predict(X_test_normalizedd, verbose=0)
test_predictions_class = np.argmax(test_predictions_prob, axis=1)

print(f"Test set shape: {X_test_normalizedd.shape}")
print(f"Predictions shape: {test_predictions_prob.shape}")
print(f"Sample predictions: {test_predictions_class[:10]}")
print(f"Prediction confidence (first 10): {np.max(test_predictions_prob[:10], axis=1)}")

# Save predictions if needed
# np.save('test_predictions.npy', test_predictions_class)

# 4. Training set evaluation (to check overfitting)
print("="*50)
print("TRAINING SET EVALUATION")
print("="*50)

train_acc, train_pred, _ = evaluate_model(model, X_train_normalized, y_train, "Training")
train_report_acc = get_classification_report(y_train, train_pred, "Training")

# 5. Plot training history
plot_training_history(history_cnn)

# 6. Summary
print("="*50)
print("FINAL SUMMARY")
print("="*50)
print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"Test Predictions Generated: {len(test_predictions_class)} samples")

# Overfitting check
overfitting = train_acc - val_acc
if overfitting > 0.1:
    print(f"🚨 Overfitting: {overfitting*100:.1f}% difference")
elif overfitting > 0.05:
    print(f"⚠️ Possible overfitting: {overfitting*100:.1f}% difference")
else:
    print(f"✅ Good model: {overfitting*100:.1f}% difference")
"""model = Sequential()


model.add(Conv2D(32, (2,2), activation='relu', input_shape=(64,64,1)))
model.add(tf.keras.layers.BatchNormalization())
model.add(MaxPooling2D((2,2)))

# Second convolutional block - removed incorrect input_shape
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(MaxPooling2D((2,2)))

# Third convolutional block
model.add(Conv2D(96, (2,2), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(MaxPooling2D((2,2)))

# Flatten and dense layers
model.add(Flatten(name='flatten'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Added dropout for regularization
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(43, activation='softmax'))

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.002),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fixed early stopping callback
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)
model.save("cnn_model.keras")

# Display model architecture
print("\n>>> Model Architecture")
model.summary()

print("\n>>> Training CNN")
history_cnn = model.fit(
    X_train_normalized, y_train,
    validation_split=0.2,
    batch_size=64,
    epochs=50,
    callbacks=[early_stop],  # Fixed callback usage
    verbose=2
)

X_test = np.array(test_edited_images)
X_test_normalizedd = X_test /255

predictions_prob = model.predict(X_test_normalizedd)

# Get the predicted class (the index of the highest probability)
predictions_class = np.argmax(predictions_prob, axis=1)

# Display predictions (or save them for later inspection)
print("Result: ",predictions_class)

# print(f"CNN  Test Accuracy: {test_acc_cnn:.4f}")

val_loss_cnn,  val_acc_cnn  = model.evaluate(X_val, y_val, verbose=0)
print(f"CNN  Valid Accuracy: {val_acc_cnn:.4f}")"""

""""""
def plot_validation_accuracy(h1):
    """Overlay validation accuracy curves for FFNN vs CNN."""
    plt.figure(figsize=(8,5))
    plt.plot(h1.history['val_accuracy'], label='FFNN Val Acc')
    plt.title('Validation Accuracy: FFNN vs CNN')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
plot_training_history(history_cnn)

# i = 0 
# for img in training_images:
#     image = cv2.imread(img)
#     image_original = cv2.resize(image, (128,128), interpolation=cv2.INTER_AREA)
#     i+=1
#     # saveimages = cv2.imwrite()

