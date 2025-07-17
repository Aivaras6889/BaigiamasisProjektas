import os
import shutil
import zipfile
import cv2
from flask import flash, redirect, url_for
import pandas as pd
import app
from werkzeug.utils import secure_filename
from app.extensions import db
from app.config import Config
from app import config
from app.models.traffic_signs import TrafficSign
from app.utils.dataset import clear_dataset
from app.utils.images import load_images_from_directory, save_image_to_dataset

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def save_uploaded_file(file):
    """Save the uploaded file to the upload folder."""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(file_path)
        return file_path, filename
    else:
        raise ValueError("File type not allowed or no file provided.")
    
def add_image_to_dataset(file, class_id, is_train=True):
    """Add an image to the dataset."""
    file_path, filename = save_uploaded_file(file)
    if not file_path:
        return False
    try:
        if not (0 <= class_id < 42):
            return False
        
        image = cv2.imread(file_path)
        if image is None:
            os.remove(file_path)
            return False
        height, width = image.shape[:2]

        sign = TrafficSign(
            filename=filename,
            image_path=file_path,
            class_id=class_id,
            width=width,
            height=height,
            is_train=is_train   
        )
        db.session.add(sign)
        db.session.commit()

        return True, f"Image added to {'training' if is_train else 'testing'} dataset successfully."
    except Exception as e:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        db.session.rollback()
        return False, f"Error adding image: {str(e)}"
    
def handle_directory_loading(form):
    dataset_path = form.dataset_path.data
    load_type = form.load_type.data
    clear_existing = form.clear_existing.data
    
    try:
        # Validate directory structure
        if not os.path.exists(dataset_path):
            flash('Dataset directory does not exist', 'error')
            return redirect(url_for('dataset.dataset_loader'))
        
        train_path = os.path.join(dataset_path, 'Train')
        test_path = os.path.join(dataset_path, 'Test')
        
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            flash('Directory must contain Train/ and Test/ subdirectories', 'error')
            return redirect(url_for('dataset.dataset_loader'))
        
        # Clear existing dataset if requested
        if clear_existing:
            clear_dataset()
        
        # Load images from directory structure
        loaded_count = load_images_from_directory(dataset_path, load_type)
        
        flash(f'Successfully loaded {loaded_count} images from dataset', 'success')
        return redirect(url_for('dataset.dataset_management'))
        
    except Exception as e:
        flash(f'Error loading dataset: {str(e)}', 'error')
        return redirect(url_for('dataset.dataset_loader'))

def handle_csv_import(form):
    try:
        csv_file = form.csv_file.data
        base_path = form.base_path.data or ''
        
        # Save uploaded CSV
        filename = secure_filename(csv_file.filename)
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        csv_file.save(csv_path)
        
        # Read and process CSV
        df = pd.read_csv(csv_path)
        required_columns = ['image_path', 'class_id', 'dataset_type']
        
        if not all(col in df.columns for col in required_columns):
            flash(f'CSV must contain columns: {", ".join(required_columns)}', 'error')
            return redirect(url_for('dataset_loader'))
        
        loaded_count = 0
        for _, row in df.iterrows():
            image_path = os.path.join(base_path, row['image_path'])
            if os.path.exists(image_path):
                save_image_to_dataset(image_path, row['class_id'], row['dataset_type'])
                loaded_count += 1
        
        flash(f'Successfully imported {loaded_count} images from CSV', 'success')
        os.remove(csv_path)  # Clean up
        return redirect(url_for('dataset_management'))
        
    except Exception as e:
        flash(f'Error importing CSV: {str(e)}', 'error')
        return redirect(url_for('dataset_loader'))

def handle_batch_upload(form):
    try:
        zip_file = form.zip_file.data
        filename = secure_filename(zip_file.filename)
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        zip_file.save(zip_path)
        
        # Extract ZIP file
        extract_path = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Find dataset directory in extracted files
        dataset_dir = find_dataset_directory(extract_path)
        if dataset_dir:
            loaded_count = load_images_from_directory(dataset_dir, 'full')
            flash(f'Successfully loaded {loaded_count} images from ZIP', 'success')
        else:
            flash('No valid dataset structure found in ZIP file', 'error')
        
        return redirect(url_for('dataset_management'))
        
    except Exception as e:
        flash(f'Error processing ZIP file: {str(e)}', 'error')
        return redirect(url_for('dataset_loader'))

def find_dataset_directory(extract_path):
    """Find valid dataset directory in extracted files"""
    for root, dirs, files in os.walk(extract_path):
        # Look for Train/ and Test/ subdirectories
        if 'Train' in dirs and 'Test' in dirs:
            return root
        # Also check for lowercase versions
        if 'train' in dirs and 'test' in dirs:
            return root
    return None

