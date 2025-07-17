import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from preprocessing import train_edited_images, test_edited_images, labels_test, labels_train

def evaluate_model(model, X_test, y_test, dataset_name="Test"):
    print(f"\n=== {dataset_name} Set Evaluation ===")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    predictions = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    sklearn_accuracy = accuracy_score(y_test, pred_classes)
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Sklearn Accuracy: {sklearn_accuracy:.4f} ({sklearn_accuracy*100:.2f}%)")
    return test_accuracy, pred_classes, predictions

def get_classification_report(y_true, y_pred, dataset_name="Test"):
    print(f"\n=== {dataset_name} Set Classification Report ===")
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred))
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title(f"{dataset_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{dataset_name.lower()}_confusion_matrix.png")
    plt.close()
    return accuracy

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(history.history['loss'], label='Training Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.close()

X = np.array(train_edited_images)
y = np.array([int(label) for label in labels_train])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train_normalized = X_train / 255.0
X_val_normalized = X_val / 255.0

model = Sequential()
model.add(Conv2D(32, (2, 2), activation='relu', input_shape=(96, 96, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(96, (2, 2), activation='relu'))
model.add(MaxPooling2D((2, 2)))
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

history_cnn = model.fit(
    X_train_normalized, y_train,
    validation_data=(X_val_normalized, y_val),
    batch_size=128,
    epochs=50,
    callbacks=[early_stop],
    verbose=2
)

model.save("cnn_model.keras")

X_test = np.array(test_edited_images)
X_test_normalized = X_test / 255.0
Y_test = np.array([int(label) for label in labels_test])

val_acc, val_pred, val_proba = evaluate_model(model, X_val_normalized, y_val, "Validation")
get_classification_report(y_val, val_pred, "Validation")

test_predictions_prob = model.predict(X_test_normalized, verbose=0)
test_predictions_class = np.argmax(test_predictions_prob, axis=1)

train_acc, train_pred, _ = evaluate_model(model, X_train_normalized, y_train, "Training")
get_classification_report(y_train, train_pred, "Training")

plot_training_history(history_cnn)

print("=" * 50)
print("FINAL SUMMARY")
print("=" * 50)
print(f"Training Accuracy: {train_acc:.4f} ({train_acc * 100:.2f}%)")
print(f"Validation Accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)")
print(f"Test Predictions Generated: {len(test_predictions_class)} samples")

overfitting = train_acc - val_acc
if overfitting > 0.1:
    print(f"Overfitting: {overfitting * 100:.1f}% difference")
elif overfitting > 0.05:
    print(f"Possible overfitting: {overfitting * 100:.1f}% difference")
else:
    print(f"Good model: {overfitting * 100:.1f}% difference")
