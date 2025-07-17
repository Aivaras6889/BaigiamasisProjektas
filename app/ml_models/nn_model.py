import os
import warnings
import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
from preprocessing import train_edited_images,test_edited_images, labels

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
y = np.array([int(label) for label in labels])
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

"""--KNeighborsClassifier model--"""

def grid_search_model(X_train, y_train, param_grid, estimator, cv, scoring, verbose, n_jobs=1):
    
    grid_search= GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, scoring=scoring,verbose=verbose, n_jobs=n_jobs)
    grid_search.fit(X_train,y_train)
    print(f"Total combinations tried: {len(grid_search.cv_results_['params'])}\n")
    for idx, params in enumerate(grid_search.cv_results_['params'], start=1):
        print(f"Variant {idx:03d}: {params}")
    
    return grid_search
def svc_model():
    pass

# knn_param_grid = {
#     'n_neighbors': [3, 5, 7, 9],
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean', 'manhattan']
# }
"""Nesuveike nes meta klaida: number of unique classes is greater than 50% of the number of samples. pasidometi pirmadieni --"""
# knn_estimator =KNeighborsClassifier()

# best_model=grid_search_model(param_grid=knn_param_grid,estimator= knn_estimator, scoring='accuracy', verbose=1)

# y_pred = best_model.predict(X_val_normalizedd_flat)
# print("Test Accuracy:", accuracy_score(y_val, y_pred))

"""SVC - Letas krovimas ir reikalauja daug resursu --"""

# svc_param_grid = [
#     # linear kernel
#     # {
#     #     'kernel': ['linear'],
#     #     'C': [0.01, 0.1, 1, 10, 100],
#     #     'class_weight': [None, 'balanced']
#     # },
#     # RBF kernel
#     {
#         'kernel': ['rbf'],
#         'C': [0.01, 0.1, 1, 10, 100],
#         'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
#         'class_weight': [None, 'balanced']
#     },
    # # Polynomial kernel
    # {
    #     'kernel': ['poly'],
    #     'C': [0.01, 0.1, 1, 10],
    #     'gamma': ['scale', 'auto', 0.001, 0.01],
    #     'degree': [2, 3, 4],
    #     'coef0': [0.0, 0.1, 0.5, 1.0],
    #     'class_weight': [None, 'balanced']
    # },
    # # Sigmoid kernel
    # {
    #     'kernel': ['sigmoid'],
    #     'C': [0.01, 0.1, 1, 10],
    #     'gamma': ['scale', 'auto', 0.001, 0.01],
    #     'coef0': [0.0, 0.1, 0.5, 1.0],
    #     'class_weight': [None, 'balanced']
    # }
# ]


# estimator = SVC(decision_function_shape='ovo')

# cv=2
# """"mean_absolute_percentage_error"""
# scoring = "accuracy"


# verbose=1


# grid_search = grid_search_model(X_train=X_train_normalized_flat, y_train=y_train, param_grid=svc_param_grid, estimator=estimator,cv=cv, scoring=scoring, verbose=verbose)
# best_model_svr = grid_search.best_estimator_

# print("Best params:", grid_search.best_params_)
# print("Best CV score:", grid_search.best_score_)

# y_pred = best_model_svr.predict(X_val_normalized)
# print("Test Accuracy:", accuracy_score(y_val, y_pred))


# train_sizes, train_scores, val_scores = learning_curve(
#     best_model_svr, X_train, y_train,
#     scoring='neg_mean_absolute_error',
#     train_sizes=np.linspace(0.1, 1.0, 5),
#     cv=5, n_jobs=1
# )

"""Bandymas Be Grid Search"""

def model_predictions(model, X_train, y_train, X_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return y_pred
"""Slow and takes to much time and i didn't get results"""
# """SVC -- Modelis veikia letai neina uzkrauti"""
# model= SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)

# model_predictions(model, X_train=X_train_normalized_flat, y_train=y_train,  X_val=X_val_normalized_flat)

# train_sizes, train_scores, val_scores = learning_curve(
#     estimator=model,
#     X=X_train_normalized_flat, y=y_train,
#     train_sizes=np.linspace(0.1, 1.0, 5),
#     scoring='accuracy',
#     cv=1, n_jobs=1
# )


# def svc_learning_curve(train_sizes, train_scores, val_scores):
#     plt.figure()
#     plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train Acc')
#     plt.plot(train_sizes, val_scores.mean(axis=1),   'o-', label='Val   Acc')
#     plt.xlabel('Training set size')
#     plt.ylabel('Accuracy')
#     plt.title('SVC Learning Curve')
#     plt.legend()
#     plt.show()

# svc_learning_curve(train_sizes ,train_scores, val_scores)

"""KNeighborsClassifier"""

model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1, weights='distance')
y_pred =model_predictions(model, X_train=X_train_normalized_flat, y_train=y_train,  X_val=X_val_normalized_flat)



ccm = confusion_matrix(y_val, y_pred)

# Print a text report (you can map classes to names if you have a dict)
knn_classification_report  =  classification_report(
    y_val,
    y_pred,
    digits=3
)
print(knn_classification_report)
i = 0
knnname = f"knn{i+1}.pkl"
joblib.dump(model, f'{knnname}.pkl')

# 1) Derive the 42 unique class labels
labels = np.unique(y)    # array([0,1,2,...,41])

# 2) Compute the confusion matrix with exactly those labels
cm = confusion_matrix(y_val, y_pred, labels=labels)

# # 3) Display it, supplying the same 42 labels
# disp = ConfusionMatrixDisplay(cm, display_labels=labels)
# fig, ax = plt.subplots(figsize=(10,10))
# disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
# plt.title("Confusion Matrix (42 classes)")
# plt.show()

# def knn_learning_curve(train_sizes, train_scores, val_scores):
#     plt.figure()
#     plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train Acc')
#     plt.plot(train_sizes, val_scores.mean(axis=1),   'o-', label='Val   Acc')
#     plt.xlabel('Training set size')
#     plt.ylabel('Accuracy')
#     plt.title('SVC Learning Curve')
#     plt.legend()
#     plt.show()

"""Error numpy._core._exceptions._ArrayMemoryError: Unable to allocate 2.10 GiB for an array with shape (17200, 16384) and data type float64"""
"""
Error numpy._core._exceptions._ArrayMemoryError: Unable to allocate 5.67 GiB for an array with shape (46440, 16384) and data type float64
"""

models = {
    'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=10,min_samples_leaf=3)
}

# 2) Common learning-curve settings
train_fracs = [0.1, 0.3, 0.5, 0.7, 1.0]   # fractions of X_train
train_frac = 0.5
cv_folds    = 2
scoring     = 'accuracy'
n_jobs      = -1

plt.figure(figsize=(8, 6))

for name, model in models.items():
    # compute learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X_train_normalized_flat, y=y_train,
        train_sizes=train_fracs,
        scoring=scoring,
        cv=cv_folds,
        n_jobs=n_jobs,
        shuffle=True,
        random_state=42
    )
    # average across folds
    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)

    # plot
    plt.plot(train_fracs, train_mean, 'o-', label=f'{name} Train')
    plt.plot(train_fracs, val_mean,   's--', label=f'{name} Val')
    predict = model.fit(X_val_normalized_flat)

    # print(f"Sample predictions: {test_predictions_class[:10]}")
    # print(f"Prediction confidence (first 10): {np.max(test_predictions_prob[:10], axis=1)}")

plt.xlabel('Fraction of training data')
plt.ylabel('Accuracy')
plt.title('Learning Curves: Model Comparison')
plt.legend(loc='best', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()

# for name, model in models.items():
#     model.save(f"{name}.keras")
i=0
for name, model in models.items():
    joblib.dump(model, f'{name}{i+1}.pkl')

"""--Slow loading on big data models--"""
# xgb_clf = XGBClassifier(
#     n_estimators=100,
#     max_depth=6,
#     learning_rate=0.1,
#     random_state=42
# )

# # Works like any sklearn classifier
# xgb_clf.fit(X_train_normalized_flat, y_train)
# predictions = xgb_clf.predict(X_train_normalized_flat)

dtrain = xgb.DMatrix(X_train_normalized_flat, label=y_train)
dvalid = xgb.DMatrix(X_val_normalized_flat, label=y_val)

# Parameters
params = {
    'objective': 'multi:softmax',  # Multi-class classification
    'num_class': len(np.unique(y)),  # Number of sign types
    'eval_metric': 'mlogloss',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}


### Veikia sckit learn xgboost uztrunka ilgai uzsikrauti ####
# Train with validation monitoring
eval_results = {}
xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dvalid, 'val')],
    evals_result=eval_results,  # This stores the progress
    verbose_eval=10
)

# Then plot the results



def plot_xgb_progress(eval_results):
    epochs = range(len(eval_results['train']['mlogloss']))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, eval_results['train']['mlogloss'], 'b-', label='Training Loss')
    plt.plot(epochs, eval_results['val']['mlogloss'], 'r-', label='Validation Loss')
    plt.title('XGBoost Training Progress')
    plt.xlabel('Boosting Round')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Convert log loss to approximate accuracy
    train_acc = 1 - np.array(eval_results['train']['mlogloss']) / max(eval_results['train']['mlogloss'])
    val_acc = 1 - np.array(eval_results['val']['mlogloss']) / max(eval_results['val']['mlogloss'])
    
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy (approx)')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy (approx)')
    plt.title('XGBoost Training Accuracy')
    plt.xlabel('Boosting Round')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    i=0
    name = f"fig{i+1}.png"
    path = os.path.join("graphs/", name)
    plt.savefig(path)
    plt.show()

    

plot_xgb_progress(eval_results)
ixgb = 0
name_xgb= f"xgb{ixgb+1}.pkl"
joblib.dump(model, name_xgb)