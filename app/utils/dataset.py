from datetime import datetime
import os
import shutil
import cv2
from flask import current_app
import numpy as np
import tensorflow as tf
from app.extensions import db, scaler
from sklearn.preprocessing import StandardScaler 
from app.models.dataset import Dataset
from app.models.dataset_images import DatasetImage
from app.models.traffic_signs import TrafficSign
from app.services.dataset_services import get_test_signs, get_training_signs, test_signs_count, training_signs_count
from app.utils.features import combine_features, deserialize_features




     

def prepare_data():
        # # train_signs = db.session.query(TrafficSign).filter(TrafficSign.is_training == True).all()
        # signst= TrafficSign.query.filter(TrafficSign.is_training==True).scalar()
        # training_signs= db.session.query(signst).all()
        
        # # test_signs = db.session.query(TrafficSign).filter(TrafficSign.is_training == False).all()
        # signsf= TrafficSign.query.filter(TrafficSign.is_training==False).scalar()
        # training_signs= db.session.query(signsf).all()
        """Signs"""
        train_signs = get_training_signs()
        test_signs = get_test_signs()

        # Extract features
        X_train, y_train = _extract_features(train_signs)
        X_test, y_test = _extract_features(test_signs)

        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

def _extract_features(signs):
    """Extract combined features from database records"""
    features = []
    labels = []
    for sign in signs:
        if sign.hog_features and sign.haar_features and sign.hue_histogram:
            # Deserialize features
            hog_feat = deserialize_features(sign.hog_features)
            haar_feat = deserialize_features(sign.haar_features)
            hue_feat = deserialize_features(sign.hue_histogram)
            
            # Combine features
            combined = combine_features({
                'hog': hog_feat,
                'haar': haar_feat,
                'hue_histogram': hue_feat
            })
            
            features.append(combined)
            labels.append(sign.class_id) 
    return np.array(features), np.array(labels)



def clear_dataset():
    """Clear all existing dataset entries"""
    # Delete all images first
    for img in DatasetImage.query.all():
        db.session.delete(img)
    db.session.commit()
    
    # Then delete datasets
    for dataset in Dataset.query.all():
        db.session.delete(dataset)
    db.session.commit()
    
def get_dataset_statistics():
    """Current dataset statistics"""
    train_count =training_signs_count()
    test_count = test_signs_count()

    train_signs = get_training_signs()
    test_signs = get_test_signs()

    train_classes ={}
    test_classes = {}

    for sign in train_signs:
        train_classes[sign.class_id] = train_classes.get(sign.class_id, 0) + 1
    
    for sign in test_signs:
        test_classes[sign.class_id] = test_classes.get(sign.class_id, 0) + 1
    
    all_classes = set(train_classes.keys()).union(set(test_classes.keys()))

    return {
        'train_count': train_count,
        'test_count': test_count,
        'train_classes': train_classes,
        'test_classes': test_classes,
        'total_classes': len(all_classes)
    }


def delete_dataset_by_id(dataset_id):
    """"Delete dataset and all associated images"""""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Delete all associated images (files and database records)
    for image in dataset.images:
        # Delete physical file
        file_path = os.path.join('static', image.image_path)
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # Delete dataset (cascade will delete DatasetImage records)
    db.session.delete(dataset)
    db.session.commit()

def validate_current_dataset():
    """Validate dataset integrity and return results"""
    results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check if we have any datasets
    dataset = Dataset.query.first()
    if not dataset:
        results['is_valid'] = False
        results['errors'].append('No dataset found')
        return results
    
    # Check each class (0-42)
    for class_id in range(43):
        train_count = DatasetImage.query.filter_by(class_id=class_id, dataset_type='train').count()
        test_count = DatasetImage.query.filter_by(class_id=class_id, dataset_type='test').count()
        
        if train_count == 0 and test_count == 0:
            results['errors'].append(f'Class {class_id} has no images')
            results['is_valid'] = False
        elif train_count == 0:
            results['warnings'].append(f'Class {class_id} has no training images')
        elif test_count == 0:
            results['warnings'].append(f'Class {class_id} has no test images')
        elif train_count < 5:
            results['warnings'].append(f'Class {class_id} has only {train_count} training images')
    
    return results

num_classes = 43


def export_current_dataset():
    """Export dataset to ZIP file"""
    import zipfile
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    export_dir = 'static/exports'
    os.makedirs(export_dir, exist_ok=True)
    
    export_filename = f"traffic_signs_dataset_{timestamp}.zip"
    export_path = os.path.join(export_dir, export_filename)
    
    with zipfile.ZipFile(export_path, 'w') as zipf:
        images = DatasetImage.query.all()
        for img in images:
            file_path = os.path.join('static', img.image_path)
            if os.path.exists(file_path):
                zipf.write(file_path, img.image_path)
    
    return export_path

def delete_current_dataset():
    """"Delete current dataset (all images and records)"""""
    # Delete all physical files
    # images = DatasetImage.query.all()
    # # for img in images:
    # #     file_path = os.path.join('static', img.image_path)
    # #     if os.path.exists(file_path):
    # #         os.remove(file_path)
    
    # Delete database records
    DatasetImage.query.delete()
    Dataset.query.delete()
    
    # Remove dataset directory
    # dataset_dir = 'static/dataset'
    # if os.path.exists(dataset_dir):
    #     shutil.rmtree(dataset_dir)
    
    db.session.commit()

def clear_dataset():
    """Clear all existing dataset entries"""
    try:
        DatasetImage.query.delete()
        Dataset.query.delete()
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        raise e
    

def get_dataset_statistics():
    """Get dataset statistics"""
    try:
        total_images = DatasetImage.query.count()
        num_classes = db.session.query(DatasetImage.class_id).distinct().count()
        train_images = DatasetImage.query.filter_by(dataset_type='train').count()
        test_images = DatasetImage.query.filter_by(dataset_type='test').count()
        
        return {
            'total_images': total_images,
            'num_classes': num_classes,
            'train_images': train_images,
            'test_images': test_images
        }
    except Exception as e:
        current_app.logger.error(f"Error getting statistics: {str(e)}")
        return None