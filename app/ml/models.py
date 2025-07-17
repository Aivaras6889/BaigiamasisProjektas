from datetime import time

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from app.models.model_result import ModelResult
from app.utils.features import combine_features, extract_all_features
from app.utils.dataset import prepare_data
from app.extensions import db ,scaler





def train_all_models(models):
    """Train all traditional ML models"""
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data()
    
    results = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'training_time': training_time
        }
        
        results[model_name] = metrics
        trained_models[model_name] = model
        
        # Save to database
        result_record = ModelResult(
            model_name=model_name,
            model_type='traditional',
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            training_time=metrics['training_time']
        )
        db.session.add(result_record)
        
        # Save model
        joblib.dump(model, f'{model_name}_model.pkl')
        
        print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}")
    
    db.session.commit()
    return results



models = {}
trained_models ={}
