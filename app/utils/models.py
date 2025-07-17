
import os
import shutil
from flask import json
import joblib
import numpy as np
import tensorflow as tf
from app.models.model_result import ModelResult
from app.models.trained_models import TrainedModel
from app.extensions import db
from werkzeug.utils import secure_filename

def get_available_models():
    """Get list of all available models"""
    models = TrainedModel.query.filter_by(is_active=True).order_by(TrainedModel.created_at.desc()).all()
    return [model.to_dict() for model in models]

def load_model_util(model_id):
    """Load model by ID from database"""
    loaded_models={}
    if model_id in loaded_models:
        return loaded_models[model_id]
    
    db_model = TrainedModel.query.get(model_id)
    if not db_model:
        raise ValueError(f"Model with ID {model_id} not found")
    
    if not db_model.is_active:
        raise ValueError(f"Model {db_model.name} is inactive")
    
    try:
        if db_model.framework == 'sklearn':
            model = joblib.load(db_model.file_path)
            scaler = None
            if db_model.scaler_path and os.path.exists(db_model.scaler_path):
                scaler = joblib.load(db_model.scaler_path)
            
            model_wrapper = {
                'model': model,
                'scaler': scaler,
                'framework': 'sklearn',
                'metadata': db_model
            }
            
        elif db_model.framework == 'tensorflow':
            model = tf.keras.models.load_model(db_model.file_path)
            
            model_wrapper = {
                'model': model,
                'scaler': None,
                'framework': 'tensorflow',
                'metadata': db_model
            }
        
        else:
            raise ValueError(f"Unsupported framework: {db_model.framework}")
        
        loaded_models[model_id] = model_wrapper
        return model_wrapper
        
    except Exception as e:
        raise ValueError(f"Failed to load model {db_model.name}: {str(e)}")
    
def detect_model_type(filename):
    """Detect both framework and model type from filename"""
    filename_lower = filename.lower()
    
    # Detect model type from filename
    if 'svm' in filename_lower:
        model_type = 'svm'
    elif 'rf' in filename_lower or 'forest' in filename_lower or 'randomforest' in filename_lower:
        model_type = 'random_forest'
    elif 'xgb' in filename_lower or 'xgboost' in filename_lower:
        model_type = 'xgboost'
    elif 'lgb' in filename_lower or 'lightgbm' in filename_lower:
        model_type = 'lightgbm'
    elif 'cat' in filename_lower or 'catboost' in filename_lower:
        model_type = 'catboost'
    elif 'nb' in filename_lower or 'naive' in filename_lower:
        model_type = 'naive_bayes'
    elif 'knn' in filename_lower:
        model_type = 'knn'
    if 'cnn' in filename_lower or 'conv' in filename_lower:
        model_type = 'cnn'
    elif 'nn' in filename_lower or 'neural' in filename_lower:
        model_type = 'neural_network' 
    elif 'logistic' in filename_lower or 'lr' in filename_lower:
        model_type = 'logistic_regression'
    else:
        model_type = 'unknown'  # Don't default to svm
    
    # Detect framework
    if filename_lower.endswith(('.joblib', '.pkl')):
        framework = 'sklearn'
    elif filename_lower.endswith(('.keras', '.h5')):
        framework = 'tensorflow'
    elif filename_lower.endswith(('.pt', '.pth')):
        framework = 'pytorch'
    else:
        framework = 'unknown'
    
    return {
        'model_type': model_type,
        'framework': framework
    }

def save_uploaded_model(form):
    """Save uploaded model file"""
    file = form.model_file.data
    filename = secure_filename(file.filename)
    model_path = os.path.join('static/models', filename)
    file.save(model_path)
    
    # Create database record
    model = TrainedModel(
        name=form.model_name.data,
        model_type=form.model_type.data,
        framework=form.framework.data,
        file_path=model_path,
        is_active=True
    )
    db.session.add(model)
    db.session.commit()
    
    return model.id

def save_directory_model(file_path, form):
    """Save model from directory"""
    filename = os.path.basename(file_path)
    target_path = os.path.join('static/models', filename)
    shutil.copy2(file_path, target_path)
    
    # Create database record
    model = TrainedModel(
        name=form.model_name.data,
        model_type=form.model_type.data,
        framework=form.framework.data,
        file_path=target_path,
        is_active=True
    )
    db.session.add(model)
    db.session.commit()
    
    return model.id



def load_model_by_id(model_id):
    """Load model by ID with support for multiple frameworks"""
    db_model = TrainedModel.query.get(model_id)
    if not db_model or not db_model.is_active:
        raise ValueError(f"Model not found or inactive")
    
    try:
        if db_model.framework == 'sklearn':
            model = joblib.load(db_model.file_path)
            scaler = None
            if db_model.scaler_path and os.path.exists(db_model.scaler_path):
                scaler = joblib.load(db_model.scaler_path)
            
            return {
                'model': model,
                'scaler': scaler,
                'framework': 'sklearn',
                'metadata': db_model
            }
            
        elif db_model.framework == 'tensorflow':
            import tensorflow as tf
            model = tf.keras.models.load_model(db_model.file_path)
            
            return {
                'model': model,
                'scaler': None,
                'framework': 'tensorflow',
                'metadata': db_model
            }
            
        elif db_model.framework == 'xgboost':
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(db_model.file_path)
            
            return {
                'model': model,
                'scaler': None,
                'framework': 'xgboost',
                'metadata': db_model
            }
            
        # elif db_model.framework == 'lightgbm':
        #     import lightgbm as lgb
        #     model = lgb.Booster(model_file=db_model.file_path)
            
        #     return {
        #         'model': model,
        #         'scaler': None,
        #         'framework': 'lightgbm',
        #         'metadata': db_model
        #     }
            
        # elif db_model.framework == 'catboost':
        #     from catboost import CatBoostClassifier
        #     model = CatBoostClassifier()
        #     model.load_model(db_model.file_path)
            
        #     return {
        #         'model': model,
        #         'scaler': None,
        #         'framework': 'catboost',
        #         'metadata': db_model
        #     }
            
        else:
            raise ValueError(f"Unsupported framework: {db_model.framework}")
            
    except Exception as e:
        raise ValueError(f"Failed to load model: {str(e)}")

def predict_with_model(model_id, features):
    """Make prediction with loaded model"""
    model_wrapper = load_model_by_id(model_id)
    model = model_wrapper['model']
    scaler = model_wrapper['scaler']
    framework = model_wrapper['framework']
    
    try:
        if framework == 'sklearn':
            if scaler:
                features_scaled = scaler.transform([features])
            else:
                features_scaled = [features]
            
            prediction = model.predict(features_scaled)[0]
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_scaled)[0]
                confidence = float(np.max(probabilities))
            else:
                confidence = None
            
            return {
                'prediction': int(prediction),
                'confidence': confidence,
                'model_name': model_wrapper['metadata'].name,
                'model_type': model_wrapper['metadata'].model_type
            }
            
        elif framework == 'tensorflow':
            features_array = np.array([features])
            predictions = model.predict(features_array)
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            
            return {
                'prediction': predicted_class,
                'confidence': confidence,
                'model_name': model_wrapper['metadata'].name,
                'model_type': model_wrapper['metadata'].model_type
            }
            
        elif framework == 'xgboost':
            import xgboost as xgb
            dtest = xgb.DMatrix([features])
            predictions = model.predict(dtest)
            
            if len(predictions.shape) > 1:  # Multi-class
                predicted_class = int(np.argmax(predictions[0]))
                confidence = float(np.max(predictions[0]))
            else:  # Binary
                predicted_class = int(predictions[0] > 0.5)
                confidence = float(predictions[0]) if predicted_class else float(1 - predictions[0])
            
            return {
                'prediction': predicted_class,
                'confidence': confidence,
                'model_name': model_wrapper['metadata'].name,
                'model_type': model_wrapper['metadata'].model_type
            }
            
        elif framework == 'lightgbm':
            predictions = model.predict([features])
            
            if len(predictions[0]) > 1:  # Multi-class
                predicted_class = int(np.argmax(predictions[0]))
                confidence = float(np.max(predictions[0]))
            else:  # Binary
                predicted_class = int(predictions[0] > 0.5)
                confidence = float(predictions[0]) if predicted_class else float(1 - predictions[0])
            
            return {
                'prediction': predicted_class,
                'confidence': confidence,
                'model_name': model_wrapper['metadata'].name,
                'model_type': model_wrapper['metadata'].model_type
            }
            
        elif framework == 'catboost':
            predictions = model.predict_proba([features])
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            
            return {
                'prediction': predicted_class,
                'confidence': confidence,
                'model_name': model_wrapper['metadata'].name,
                'model_type': model_wrapper['metadata'].model_type
            }
            
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")

def save_trained_model(model, framework, model_name, model_type, metrics=None, hyperparameters=None, scaler=None):
    """Save model to database and filesystem"""
    models_dir = 'static/models'
    model_dir = os.path.join(models_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    if framework == 'sklearn':
        model_path = os.path.join(model_dir, 'model.joblib')
        joblib.dump(model, model_path)
        
        scaler_path = None
        if scaler:
            scaler_path = os.path.join(model_dir, 'scaler.joblib')
            joblib.dump(scaler, scaler_path)
            
    elif framework == 'tensorflow':
        model_path = os.path.join(model_dir, 'model.keras')
        model.save(model_path)
        scaler_path = None
        
    elif framework == 'xgboost':
        model_path = os.path.join(model_dir, 'model.json')
        model.save_model(model_path)
        scaler_path = None
        
    elif framework == 'lightgbm':
        model_path = os.path.join(model_dir, 'model.txt')
        model.save_model(model_path)
        scaler_path = None
        
    elif framework == 'catboost':
        model_path = os.path.join(model_dir, 'model.cbm')
        model.save_model(model_path)
        scaler_path = None
    
    # Save to database
    db_model = TrainedModel(
        name=model_name,
        model_type=model_type,
        framework=framework,
        file_path=model_path,
        scaler_path=scaler_path,
        accuracy=metrics.get('accuracy') if metrics else None,
        precision=metrics.get('precision') if metrics else None,
        recall=metrics.get('recall') if metrics else None,
        f1_score=metrics.get('f1_score') if metrics else None,
        training_samples=metrics.get('training_samples') if metrics else None,
        training_time=metrics.get('training_time') if metrics else None,
        hyperparameters=json.dumps(hyperparameters) if hyperparameters else None
    )
    
    db.session.add(db_model)
    db.session.commit()
    
    return db_model.id

def get_available_models():
    """Get list of all available models"""
    return TrainedModel.query.filter_by(is_active=True).order_by(TrainedModel.created_at.desc()).all()

def extract_features_from_image(image_path):
    """Extract features from image for ML prediction"""
    # Your existing feature extraction code
    import numpy as np
    from skimage import io, feature, color, transform
    
    # Load image
    image = io.imread(image_path)
    
    # Convert to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_gray = color.rgb2gray(image)
    else:
        image_gray = image
    
    # Resize to standard size
    image_resized = transform.resize(image_gray, (32, 32))
    
    # Extract HOG features
    hog_features = feature.hog(
        image_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )
    
    # Extract color features if RGB image
    if len(image.shape) == 3:
        image_rgb = transform.resize(image, (32, 32))
        color_features = [
            np.mean(image_rgb[:, :, 0]),  # Mean Red
            np.mean(image_rgb[:, :, 1]),  # Mean Green
            np.mean(image_rgb[:, :, 2]),  # Mean Blue
            np.std(image_rgb[:, :, 0]),   # Std Red
            np.std(image_rgb[:, :, 1]),   # Std Green
            np.std(image_rgb[:, :, 2])    # Std Blue
        ]
    else:
        color_features = [0] * 6
    
    # Combine features
    features = np.concatenate([hog_features, color_features])
    return features.tolist()



def calculate_model_performance(model_id):
    """Calculate real performance from actual predictions"""
    from app.models import Results
    
    # Get all predictions for this model where we know actual class
    predictions = db.session.query(Results).filter(
        Results.model_name.like(f'%{model_id}%'),
        Results.actual_class.isnot(None)
    ).all()
    
    if not predictions:
        return None
    
    correct = 0
    total = len(predictions)
    confidences = []
    
    for pred in predictions:
        if int(pred.prediction) == pred.actual_class:
            correct += 1
        if pred.confidence:
            confidences.append(pred.confidence)
    
    accuracy = correct / total if total > 0 else 0
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    return {
        'accuracy': accuracy,
        'total_predictions': total,
        'correct_predictions': correct,
        'avg_confidence': avg_confidence
    }



def save_model_performance(model_id):
    """Save real model performance to ModelResult table"""
    model = TrainedModel.query.get(model_id)
    if not model:
        return False
    
    performance = calculate_model_performance(model_id)
    if not performance:
        return False
    
    # Create or update ModelResult
    existing = ModelResult.query.filter_by(model_name=model.name).first()
    if existing:
        existing.accuracy = performance['accuracy']
        existing.precision = performance['accuracy']  # Simplified
        existing.recall = performance['accuracy']     # Simplified  
        existing.f1_score = performance['accuracy']   # Simplified
    else:
        result = ModelResult(
            model_name=model.name,
            model_type=model.model_type,
            accuracy=performance['accuracy'],
            precision=performance['accuracy'],  # You can calculate proper precision/recall if needed
            recall=performance['accuracy'],
            f1_score=performance['accuracy'],
            training_time=0,  # Unknown for uploaded models
            hyperparameters=f"Total predictions: {performance['total_predictions']}"
        )
        db.session.add(result)
    
    db.session.commit()
    return True