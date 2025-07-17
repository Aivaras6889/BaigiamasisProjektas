from typing import Dict, List
from sklearn.model_selection import ParameterGrid
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
import pandas as pd
import cv2
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import itertools
from app.models.traffic_signs import TrafficSign
from app.models.model_result import ModelResult
from app.extensions import db
from app.utils.images import prepare_images
# class NeuralNetworkModels:
#     def __init__(self, session):
#         self.session = session
#         self.input_shape = (64, 64, 3)
#         self.num_classes = 43  # Classes 0-42
#         self.hyperparameter_results = []



def create_cnn_model(params, input_shape, num_classes):
    """Create CNN model with given hyperparameters"""
    model = models.Sequential()
    
    # Conv layers
    model.add(layers.Conv2D(params['conv1_filters'], (params['conv1_kernel'], params['conv1_kernel']),
                            activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(params['conv2_filters'], (params['conv2_kernel'], params['conv2_kernel']),
                            activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    if params['num_conv_layers'] >= 3:
        model.add(layers.Conv2D(params['conv3_filters'], (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
    
    # Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(params['dense1_units'], activation='relu'))
    model.add(layers.Dropout(params['dropout1']))
    
    if params['num_dense_layers'] >= 2:
        model.add(layers.Dense(params['dense2_units'], activation='relu'))
        model.add(layers.Dropout(params['dropout2']))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# def generate_hyperparameter_combinations(self):
    """Generate 30+ hyperparameter combinations"""
    # Define parameter grids
    param_grid = {
        'conv1_filters': [32, 64],
        'conv1_kernel': [3, 5],
        'conv2_filters': [64, 128],
        'conv2_kernel': [3, 5],
        'conv3_filters': [128, 256],
        'num_conv_layers': [2, 3],
        'dense1_units': [128, 256, 512],
        'dense2_units': [64, 128],
        'num_dense_layers': [1, 2],
        'dropout1': [0.3, 0.5],
        'dropout2': [0.3, 0.5],
        'learning_rate': [0.001, 0.0001],
        'batch_size': [32, 64],
        'optimizer': ['adam', 'sgd'],
        'epochs': [20, 30]
    }
    
    # Generate systematic combinations
    combinations = []
    
    # First, create base combinations
    for conv1_f in param_grid['conv1_filters']:
        for conv2_f in param_grid['conv2_filters']:
            for dense1_u in param_grid['dense1_units']:
                for lr in param_grid['learning_rate']:
                    for bs in param_grid['batch_size']:
                        for opt in param_grid['optimizer']:
                            combination = {
                                'conv1_filters': conv1_f,
                                'conv1_kernel': 3,
                                'conv2_filters': conv2_f,
                                'conv2_kernel': 3,
                                'conv3_filters': 128,
                                'num_conv_layers': 2,
                                'dense1_units': dense1_u,
                                'dense2_units': 128,
                                'num_dense_layers': 1,
                                'dropout1': 0.5,
                                'dropout2': 0.5,
                                'learning_rate': lr,
                                'batch_size': bs,
                                'optimizer': opt,
                                'epochs': 20
                            }
                            combinations.append(combination)
                            
                            if len(combinations) >= 30:
                                return combinations[:30]
    
    return combinations[:30]  # Ensure exactly 30 combinations

def generate_sklearn_style(max_combinations: int = 30) -> List[Dict]:
   
    # Define parameter grid
    param_grid = {
        'conv1_filters': [16, 32, 64],
        'conv2_filters': [32, 64, 128], 
        'dense1_units': [128, 256, 512],
        'learning_rate': [0.001, 0.0001],
        'batch_size': [32, 64],
        'optimizer': ['adam', 'sgd', 'rmsprop']
    }
    
    # Fixed parameters
    fixed_params = {
        'conv1_kernel': 3,
        'conv2_kernel': 3,
        'conv3_filters': 128,
        'conv3_kernel': 3,
        'num_conv_layers': 2,
        'dense2_units': 128,
        'num_dense_layers': 1,
        'dropout1': 0.5,
        'dropout2': 0.5,
        'epochs': 20
    }
    
    # Generate combinations
    grid = ParameterGrid(param_grid)
    combinations = []
    
    for params in grid:
        # Combine with fixed parameters
        full_params = {**fixed_params, **params}
        combinations.append(full_params)
        
        if len(combinations) >= max_combinations:
            break
    
    return combinations[:max_combinations]



def train_with_hyperparameters( X_train, X_test, y_train, y_test, params):
    """Train model with specific hyperparameters"""
    try:
        # Create model
        model = create_cnn_model(params)
        
        # Compile
        optimizer = optimizers.Adam(learning_rate=params['learning_rate']) if params['optimizer'] == 'adam' else optimizers.SGD(learning_rate=params['learning_rate'])
        
        model.compile(optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Train
        start_time = time.time()
        history = model.fit(X_train, y_train,
                            batch_size=params['batch_size'],
                            epochs=params['epochs'],
                            validation_data=(X_test, y_test),
                            callbacks=[early_stopping],
                            verbose=0)
        
        training_time = time.time() - start_time
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # Calculate additional metrics
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        precision = precision_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
        recall = recall_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
        
        result = {
            'hyperparams': params,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': training_time,
            'epochs_trained': len(history.history['loss'])
        }
        
        return result, model
        
    except Exception as e:
        print(f"Error training with params {params}: {e}")
        return None, None

def comprehensive_hyperparameter_testing():
    """Run comprehensive hyperparameter testing (30+ experiments)"""
    print("Starting comprehensive hyperparameter testing...")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_images()
    
    # Generate combinations
    combinations = generate_sklearn_style()
    print(f"Testing {len(combinations)} hyperparameter combinations")
    
    results = []
    best_model = None
    best_accuracy = 0
    
    for i, params in enumerate(combinations):
        print(f"Testing combination {i+1}/{len(combinations)}")
        
        result, model = train_with_hyperparameters(X_train, X_test, y_train, y_test, params)
        
        if result:
            results.append(result)
            
            # Save to database
            result_record = ModelResult(
                model_name=f"neural_network_{i+1}",
                model_type='neural',
                accuracy=result['test_accuracy'],
                precision=result['precision'],
                recall=result['recall'],
                f1_score=result['f1_score'],
                training_time=result['training_time'],
                hyperparameters=str(result['hyperparams'])
            )
            db.session.add(result_record)
            
            # Track best model
            if result['test_accuracy'] > best_accuracy:
                best_accuracy = result['test_accuracy']
                best_model = model
                model.save('best_neural_network.h5')
            
            print(f"  Accuracy: {result['test_accuracy']:.4f}")
    
    db.session.commit()
    
    # Create results DataFrame
    results_df = _create_results_dataframe(results)
    results_df.to_csv('hyperparameter_results.csv', index=False)
    
    print(f"Testing completed. Best accuracy: {best_accuracy:.4f}")
    return results_df, best_model

def _create_results_dataframe(results):
    """Create DataFrame from results"""
    data = []
    for i, result in enumerate(results):
        row = {
            'Experiment': i + 1,
            'Test_Accuracy': result['test_accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1_Score': result['f1_score'],
            'Training_Time': result['training_time'],
            'Epochs_Trained': result['epochs_trained']
        }
        
        # Add hyperparameters
        for key, value in result['hyperparams'].items():
            row[key] = value
        
        data.append(row)
    
    return pd.DataFrame(data)