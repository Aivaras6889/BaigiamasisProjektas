import datetime
from imblearn.over_sampling import SMOTE
import os
import joblib
from matplotlib import pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
import xgboost
from preprocessing import train_edited_images, labels_train, test_edited_images,labels_test


def grid_search_model(X_train, y_train, param_grid, estimator, cv, scoring, verbose, n_jobs=1):
    
    grid_search= GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, scoring=scoring,verbose=verbose, n_jobs=n_jobs)
    grid_search.fit(X_train,y_train)
    print(f"Total combinations tried: {len(grid_search.cv_results_['params'])}\n")
    for idx, params in enumerate(grid_search.cv_results_['params'], start=1):
        print(f"Variant {idx:03d}: {params}")
    
    return grid_search
def svc_model():
    pass

def model_predictions(model, X_train, y_train, X_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return y_pred




cattegories = list(range(0,43))

# numeric_labels = [int(value) for value in cleaned_labels]
X = np.array(train_edited_images)
y = np.array([int(label) for label in labels_train])
print(y.dtype)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_train_normalized = X_train / 255.0  # normalized training images
X_val_normalized = X_val / 255.0  # normalized validation images
print(len(test_edited_images))
X_test = np.array(test_edited_images)
X_test_normalized = X_test / 255.0
print(X_test.shape)

X_train_normalized_flat = X_train_normalized.reshape(X_train_normalized.shape[0], -1)
X_val_normalized_flat = X_val_normalized.reshape(X_val_normalized.shape[0], -1)
X_test_normalized_flat = X_test_normalized.reshape(X_test_normalized.shape[0], -1)

y_test =  np.array([int(label) for label in labels_test])


smote = SMOTE(random_state=42, k_neighbors=3)
X_train_smote, y_train_smote = smote.fit_resample(X_train_normalized_flat, y_train)

"""--KNeighborsClassifier model--"""



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

# """KNeighborsClassifier Letas neaisku ar del duomenu ar parametru problemos""" 

# def knn_learning_curve(train_sizes, train_scores, val_scores):
#     plt.figure()
#     plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train Acc')
#     plt.plot(train_sizes, val_scores.mean(axis=1),   'o-', label='Val   Acc')
#     plt.xlabel('Training set size')
#     plt.ylabel('Accuracy')
#     plt.title('KNNC Learning Curve')
#     plt.legend()
#     plt.show()

# model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1, weights='distance', leaf_size=10, p=1)
# y_pred = model_predictions(model, X_train=X_train_smote, y_train=y_train_smote, X_val=X_val_normalized_flat)
# y_pred_test = model_predictions(model, X_train=X_train_smote, y_train=y_train_smote, X_val=X_test_normalized_flat)
# print("KNNC Validation Confusion Matrix")
# ccm = confusion_matrix(y_val, y_pred)

# # Print a text report (you can map classes to names if you have a dict)
# print("KNNC Classifaction report")
# knn_classification_report  =  classification_report(
#     y_val,
#     y_pred,
#     digits=3
# )
# print(knn_classification_report)

# # 1) Derive the 42 unique class labels
# labels_train = np.unique(y)    # array([0,1,2,...,41])
# labels_test = np.unique(y_test)
# # 2) Compute the confusion matrix with exactly those labels
# print(" KNC Validation Confusion Matrix")
# cmv = confusion_matrix(y_val, y_pred, labels=labels_train)
# print(cmv)
# print(" KNC Test Confusion Matrix")
# cmt = confusion_matrix(y_test, y_pred_test, labels=labels_test)
# print(cmt)
# print("KNNC Test Classification Report")
# ccmt = classification_report(y_test, y_pred_test, labels=labels_test)
# print(ccmt)

# # Compute learning curve data for KNNC
# train_sizes, train_scores, val_scores = learning_curve(
#     estimator=model,
#     X=X_train_smote,
#     y=y_train_smote,
#     train_sizes=np.linspace(0.3, 1.0, 5),
#     scoring='accuracy',
#     cv=3,
#     n_jobs=-1,
#     shuffle=True,
#     random_state=42
# )

# # Plot the learning curve
# knn_learning_curve(train_sizes, train_scores,val_scores)

# i = 0
# knnname = f"knn{i+1}"
# joblib.dump(model, f'app/ml_models.{knnname, i+1}.pkl')


# # 3) Display it, supplying the same 42 labels
# disp = ConfusionMatrixDisplay(cm, display_labels=labels)
# fig, ax = plt.subplots(figsize=(10,10))
# disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
# plt.title("Confusion Matrix (42 classes)")
# plt.show()




"""Error numpy._core._exceptions._ArrayMemoryError: Unable to allocate 2.10 GiB for an array with shape (17200, 16384) and data type float64"""
"""
Error numpy._core._exceptions._ArrayMemoryError: Unable to allocate 5.67 GiB for an array with shape (46440, 16384) and data type float64
"""
# antras bandymas n_estimators=100, max_depth=15, min_samples_leaf=6, n_jobs=-1, verbose=1
# pirmo bandymo parametrai n_estimators=100, max_depth=15, min_samples_leaf=6, n_jobs=-1, verbose=1
# model_rf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=6, n_jobs=-1, verbose=1)
# i=0
# name="Random Forest"
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# namerf = f"RandomForest_{timestamp}"

# train_fracs = [0.1, 0.3, 0.5, 0.7, 1.0]
# cv_folds = 3
# scoring = 'accuracy'
# n_jobs = -1

# plt.figure(figsize=(8, 6))

# # Get learning curve data
# train_sizes, train_scores, val_scores = learning_curve(
#     estimator=model_rf,
#     X=X_train_normalized_flat, y=y_train,
#     train_sizes=train_fracs,
#     scoring=scoring,
#     cv=cv_folds,
#     n_jobs=n_jobs,
#     shuffle=True,
#     random_state=42
# )

# train_mean = train_scores.mean(axis=1)
# val_mean = val_scores.mean(axis=1)

# # Compute test accuracy for each training size
# test_scores = []
# for size in train_sizes:
#     model_rf.fit(X_train_normalized_flat[:size], y_train[:size])
#     test_acc = model_rf.score(X_test_normalized_flat, y_test)
#     test_scores.append(test_acc)
# test_scores = np.array(test_scores)

# # Plot all three curves
# plt.plot(train_sizes, train_mean, 'o-', label=f'{name} Train')
# plt.plot(train_sizes, val_mean, 's--', label=f'{name} Val')
# plt.plot(train_sizes, test_scores, 'd-.', label=f'{name} Test')

# plt.xlabel('Number of training samples')
# plt.ylabel('Accuracy')
# plt.title('Learning Curves: Model Comparison')
# plt.legend(loc='best', fontsize='small')
# plt.grid(True)
# plt.tight_layout()
# plot_path = f'app/ml_models/{namerf}_learning_curve.png'
# plt.savefig(plot_path)
# plt.show()
# # for name, model in models.items():
# #     model.save(f"{name}.keras")

# # Fit the model on the full training set
# model_rf.fit(X_train_normalized_flat, y_train)

# # Predict on validation and test sets
# y_val_pred = model_rf.predict(X_val_normalized_flat)
# y_test_pred = model_rf.predict(X_test_normalized_flat)

# print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
# print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

# # Classification report
# print("Validation Classification Report:")
# print(classification_report(y_val, y_val_pred, digits=3))
# print("Test Classification Report:")
# print(classification_report(y_test, y_test_pred, digits=3))

# # Confusion matrix
# print("Validation Confusion Matrix:")
# print(confusion_matrix(y_val, y_val_pred))
# print("Test Confusion Matrix:")
# print(confusion_matrix(y_test, y_test_pred))


# joblib.dump(model_rf, f'app/ml_models/{namerf}.pkl')

# """--Slow loading on big data models--"""
# # xgb_clf = XGBClassifier(
# #     n_estimators=100,
# #     max_depth=6,
# #     learning_rate=0.1,
# #     random_state=42
# # )

# # # Works like any sklearn classifier
# # xgb_clf.fit(X_train_normalized_flat, y_train)
# # predictions = xgb_clf.predict(X_train_normalized_flat)

dtrain = xgboost.DMatrix(X_train_normalized_flat, label=y_train)
dvalid = xgboost.DMatrix(X_val_normalized_flat, label=y_val)
dtest= xgboost.DMatrix(X_test_normalized_flat)


# # Parameters
# params = {
#     'objective': 'multi:softmax',  # Multi-class classification
#     'num_class': len(np.unique(y)),  # Number of sign types
#     'eval_metric': 'mlogloss',
#     'max_depth': 6,
#     'learning_rate': 0.1,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#     'random_state': 42
# }


# ### Veikia sckit learn xgboost uztrunka ilgai uzsikrauti ####
# # Train with validation monitoring
# eval_results = {}
# xgb_model = xgboost.train(
#     params,
#     dtrain,
#     num_boost_round=100,
#     evals=[(dtrain, 'train'), (dvalid, 'val')],
#     evals_result=eval_results,  # This stores the progress
#     verbose_eval=10
# )

# # Then plot the results



# def plot_xgb_progress(eval_results):
#     epochs = range(len(eval_results['train']['mlogloss']))
    
#     plt.figure(figsize=(12, 5))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, eval_results['train']['mlogloss'], 'b-', label='Training Loss')
#     plt.plot(epochs, eval_results['val']['mlogloss'], 'r-', label='Validation Loss')
#     plt.title('XGBoost Training Progress')
#     plt.xlabel('Boosting Round')
#     plt.ylabel('Log Loss')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     plt.subplot(1, 2, 2)
#     # Convert log loss to approximate accuracy
#     train_acc = 1 - np.array(eval_results['train']['mlogloss']) / max(eval_results['train']['mlogloss'])
#     val_acc = 1 - np.array(eval_results['val']['mlogloss']) / max(eval_results['val']['mlogloss'])
    
#     plt.plot(epochs, train_acc, 'b-', label='Training Accuracy (approx)')
#     plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy (approx)')
#     plt.title('XGBoost Training Accuracy')
#     plt.xlabel('Boosting Round')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     i=0
#     name = f"fig{i+1}.png"
#     path = os.path.join("graphs/", name)
#     plt.savefig(path)
#     plt.show()

    

# plot_xgb_progress(eval_results)
# ixgb = 0
# name_xgb= f"xgb{ixgb+1}.pkl"
# joblib.dump(model, name_xgb)


# Parameters
params = {
    'objective': 'multi:softmax',
    'num_class': len(np.unique(y)),
    'eval_metric': 'mlogloss',
    'max_depth': 8,
    'learning_rate': 0.2,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# Train with validation monitoring
eval_results = {}
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
name_xgb = f"xgboost_{timestamp}"

xgb_model = xgboost.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dvalid, 'val')],
    evals_result=eval_results,
    verbose_eval=10
)

# Save the XGBoost model
xgb_model.save_model(f'app/ml_models/{name_xgb}.json')

def plot_xgb_progress(eval_results, name_xgb):
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
    # Convert log loss to approximate accuracy (optional, for visualization)
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
    plot_path = f'app/ml_models/{name_xgb}_training_progress.png'
    plt.savefig(plot_path)
    plt.show()

plot_xgb_progress(eval_results, name_xgb)


# Predict on validation and test sets
y_val_pred = xgb_model.predict(dvalid)
y_test_pred = xgb_model.predict(dtest)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred, digits=3))
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred, digits=3))
print("Validation Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))
print("Test Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))