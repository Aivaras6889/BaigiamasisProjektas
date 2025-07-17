

import os
import cv2
import joblib
import numpy as np
import tensorflow as tf

from app.models.predictions import Prediction
from app.services.predictions import get_predictions
from app.utils.features import combine_features, extract_all_features
from app.extensions import scaler,db
from app.utils.handlers import save_uploaded_file
from app.utils.images import extract_features_from_image
from app.utils.models import load_model_by_id, load_model_util

def predict_single_by_features(trained_models, image_path, model_name='svm'):
    """Predict single image"""
    if model_name not in trained_models:
        # Try to load model
        try:
            trained_models[model_name] = joblib.load(f'{model_name}_model.pkl')
        except:
            return None
    
    # Extract features
    features = extract_all_features(image_path)
    if not features:
        return None
    
    combined = combine_features(features)
    combined_scaled = scaler.transform([combined])
    
    # Predict
    prediction = trained_models[model_name].predict(combined_scaled)[0]
    
    # Get confidence if available
    confidence = None
    if hasattr(trained_models[model_name], 'predict_proba'):
        proba = trained_models[model_name].predict_proba(combined_scaled)[0]
        confidence = np.max(proba)
    
    return {
        'predicted_class': int(prediction),
        'confidence': float(confidence) if confidence else None,
        'model_used': model_name
    }


def predict_single_image(file, model_type='neural', model_name='best_neural_network'):
        """Predict class for single uploaded image"""
        filepath, filename = save_uploaded_file(file)
        
        if not filepath:
            return None, "Invalid file type"
        
        try:
            if model_type == 'neural':
                result = _predict_with_neural_network(filepath, model_name)
            else:
                result = _predict_with_traditional_ml(filepath, model_name)
            
            if result:
                # Save prediction to database
                prediction = Prediction(
                    image_path=filepath,
                    predicted_class=result['predicted_class'],
                    confidence=result.get('confidence'),
                    model_used=f"{model_type}_{model_name}"
                )
                db.session.add(prediction)
                db.session.commit()
                
                result['image_path'] = filepath
                result['filename'] = filename
            
            return result, None
            
        except Exception as e:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
            return None, f"Prediction error: {str(e)}"
        
def _predict_with_neural_network(image_path, model_name):
    """Predict using neural network"""
    try:
        # Load model
        model_path = f"{model_name}.h5"
        if not os.path.exists(model_path):
            model_path = "best_neural_network.h5"
        
        if not os.path.exists(model_path):
            raise Exception("Neural network model not found")
        
        model = tf.keras.models.load_model(model_path)
        
        # Preprocess image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (64, 64))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Predict
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        return {
            'predicted_class': int(predicted_class),
            'confidence': confidence,
            'model_used': model_name
        }
    except Exception as e:
        raise Exception(f"Neural network prediction failed: {str(e)}")
    
def _predict_with_traditional_ml(self, image_path, model_name):
    """Predict using traditional ML model"""
    try:
        # Load model and scaler
        model_path = f"{model_name}_model.pkl"
        if not os.path.exists(model_path):
            raise Exception(f"Traditional ML model {model_name} not found")
        
        model = joblib.load(model_path)
        
        # Extract features
        features = extract_all_features(image_path)
        if not features:
            raise Exception("Could not extract features from image")
        
        combined = combine_features(features)
        
        # Load scaler (assuming it's saved separately)
        scaler_path = f"{model_name}_scaler.pkl"
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            combined = scaler.transform([combined])
        else:
            combined = [combined]
        
        # Predict
        prediction = model.predict(combined)[0]
        
        # Get confidence if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(combined)[0]
            confidence = float(np.max(proba))
        
        return {
            'predicted_class': int(prediction),
            'confidence': confidence,
            'model_used': model_name
        }
        
    except Exception as e:
        raise Exception(f"Traditional ML prediction failed: {str(e)}")

def get_recent_predictions(limit=10):
    """Get recent predictions"""
    predictions = get_predictions(limit)
    return [{
        'id': pred.id,
        'predicted_class': pred.predicted_class,
        'confidence': pred.confidence,
        'model_used': pred.model_used,
        'created_at': pred.created_at.isoformat()
    } for pred in predictions]


def predict_with_model(model_id, features_or_image_path):
    """Make prediction with loaded model"""
    model_wrapper = load_model_by_id(model_id)
    model = model_wrapper['model']
    scaler = model_wrapper['scaler']
    framework = model_wrapper['framework']
    
    try:
        if framework == 'sklearn':
            # Use HOG features for sklearn models
            if isinstance(features_or_image_path, str):
                # If image path is passed, extract features
                features = extract_features_from_image(features_or_image_path)
            else:
                # Features already extracted
                features = features_or_image_path
                
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
            # For TensorFlow models, use raw image data
            if isinstance(features_or_image_path, list):
                # If features are passed, we need the original image path
                raise ValueError("TensorFlow models require image path, not extracted features")
            
            image_path = features_or_image_path
            
            # Load and preprocess image for neural network
            from skimage import io, transform, color
            import numpy as np
            
            image = io.imread(image_path)
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    image = image[:, :, :3]  # Remove alpha channel
                image = color.rgb2gray(image)
            
            # Resize to expected input size (96x96 based on error)
            image_resized = transform.resize(image, (96, 96))
            
            # Add batch and channel dimensions: (96, 96) -> (1, 96, 96, 1)
            image_array = np.expand_dims(image_resized, axis=0)  # Add batch dimension
            image_array = np.expand_dims(image_array, axis=-1)   # Add channel dimension
            
            predictions = model.predict(image_array)
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            
            return {
                'prediction': predicted_class,
                'confidence': confidence,
                'model_name': model_wrapper['metadata'].name,
                'model_type': model_wrapper['metadata'].model_type
            }
            
        elif framework == 'pytorch':
            # For PyTorch models
            if isinstance(features_or_image_path, str):
                # Load and preprocess image
                from skimage import io, transform, color
                import torch
                
                image = io.imread(features_or_image_path)
                if len(image.shape) == 3:
                    image = color.rgb2gray(image)
                
                image_resized = transform.resize(image, (96, 96))
                image_tensor = torch.FloatTensor(image_resized).unsqueeze(0).unsqueeze(0)
                
                model.eval()
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = int(torch.argmax(probabilities, dim=1)[0])
                    confidence = float(torch.max(probabilities)[0])
            else:
                # Use features for PyTorch
                import torch
                features = features_or_image_path
                
                model.eval()
                with torch.no_grad():
                    features_tensor = torch.FloatTensor([features])
                    outputs = model(features_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = int(torch.argmax(probabilities, dim=1)[0])
                    confidence = float(torch.max(probabilities)[0])
            
            return {
                'prediction': predicted_class,
                'confidence': confidence,
                'model_name': model_wrapper['metadata'].name,
                'model_type': model_wrapper['metadata'].model_type
            }
            
        elif framework == 'xgboost':
            # Use HOG features for XGBoost
            if isinstance(features_or_image_path, str):
                features = extract_features_from_image(features_or_image_path)
            else:
                features = features_or_image_path
                
            import xgboost as xgb
            dtest = xgb.DMatrix([features])
            predictions = model.predict(dtest)
            
            if len(predictions.shape) > 1:
                predicted_class = int(np.argmax(predictions[0]))
                confidence = float(np.max(predictions[0]))
            else:
                predicted_class = int(predictions[0] > 0.5)
                confidence = float(predictions[0]) if predicted_class else float(1 - predictions[0])
            
            return {
                'prediction': predicted_class,
                'confidence': confidence,
                'model_name': model_wrapper['metadata'].name,
                'model_type': model_wrapper['metadata'].model_type
            }
            
        elif framework == 'lightgbm':
            # Use HOG features for LightGBM
            if isinstance(features_or_image_path, str):
                features = extract_features_from_image(features_or_image_path)
            else:
                features = features_or_image_path
                
            predictions = model.predict([features])
            
            if len(predictions[0]) > 1:
                predicted_class = int(np.argmax(predictions[0]))
                confidence = float(np.max(predictions[0]))
            else:
                predicted_class = int(predictions[0] > 0.5)
                confidence = float(predictions[0]) if predicted_class else float(1 - predictions[0])
            
            return {
                'prediction': predicted_class,
                'confidence': confidence,
                'model_name': model_wrapper['metadata'].name,
                'model_type': model_wrapper['metadata'].model_type
            }
            
        elif framework == 'catboost':
            # Use HOG features for CatBoost
            if isinstance(features_or_image_path, str):
                features = extract_features_from_image(features_or_image_path)
            else:
                features = features_or_image_path
                
            predictions = model.predict_proba([features])
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            
            return {
                'prediction': predicted_class,
                'confidence': confidence,
                'model_name': model_wrapper['metadata'].name,
                'model_type': model_wrapper['metadata'].model_type
            }
            
        else:
            raise ValueError(f"Unsupported framework: {framework}")
            
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")