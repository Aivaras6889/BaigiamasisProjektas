import pickle
import cv2
import numpy as np
from skimage.feature import hog
from app.extensions import db

def extract_hog_features(image):
    """Extract HOG features"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = cv2.resize(gray, (64, 64))
    
    features = hog(gray, 
                  orientations=9,
                  pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2),
                  block_norm='L2-Hys')
    return features

def extract_haar_features(image):
    """Extract Haar-like features using integral image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = cv2.resize(gray, (32, 32))
    
    # Calculate integral image
    integral = cv2.integral(gray)
    
    features = []
    h, w = gray.shape
    step = 4
    
    # Extract rectangular features
    for i in range(0, h - step, step):
        for j in range(0, w - step, step):
            # Two-rectangle features (horizontal)
            if i + 2 * step < h:
                top = rect_sum(integral, i, j, step, step)
                bottom = rect_sum(integral, i + step, j, step, step)
                features.append(top - bottom)
            
            # Two-rectangle features (vertical)
            if j + 2 * step < w:
                left = rect_sum(integral, i, j, step, step)
                right = rect_sum(integral, i, j + step, step, step)
                features.append(left - right)
    
    return np.array(features)

def rect_sum(integral, x, y, w, h):
    """Calculate rectangle sum using integral image"""
    try:
        return (integral[x + h, y + w] - integral[x, y + w] - 
               integral[x + h, y] + integral[x, y])
    except IndexError:
        return 0

def extract_hue_histogram(image):
    """Extract hue histogram features"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    
    # Calculate histogram
    hist = cv2.calcHist([hue], [0], None, [180], [0, 180])
    hist = hist.flatten()
    hist = hist / (hist.sum() + 1e-6)  # Normalize
    
    return hist

def extract_all_features(image_path):
    """Extract all feature types from image"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    features = {
        'hog': extract_hog_features(image),
        'haar': extract_haar_features(image),
        'hue_histogram': extract_hue_histogram(image)
    }
    
    return features

def combine_features(features_dict):
    """Combine all features into single vector"""
    combined = []
    for feature_type in ['hog', 'haar', 'hue_histogram']:
        if feature_type in features_dict:
            combined.extend(features_dict[feature_type])
    return np.array(combined)

def serialize_features(features):
    """Serialize features for database storage"""
    return pickle.dumps(features)

def deserialize_features(serialized_features):
    """Deserialize features from database"""
    return pickle.loads(serialized_features)

# Function to update database with features
def extract_and_store_features(session):
    """Extract features for all images in database"""
    from models import TrafficSign
    
    signs = session.query(TrafficSign).all()
    
    for sign in signs:
        if sign.hog_features is None:  # Only process if not already done
            features = extract_all_features(sign.image_path)
            
            if features:
                sign.hog_features = serialize_features(features['hog'])
                sign.haar_features = serialize_features(features['haar'])
                sign.hue_histogram = serialize_features(features['hue_histogram'])
        
        if len([s for s in signs if s.hog_features is not None]) % 100 == 0:
            session.commit()
            print(f"Processed {len([s for s in signs if s.hog_features is not None])} images...")
    
    session.commit()
    print("Feature extraction completed!")