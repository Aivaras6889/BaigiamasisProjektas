# #upload images to the server
# from flask import current_app
# from werkzeug.utils import secure_filename
# import os
# from PIL import Image
# def save_image(file):
#     """Save an image file to the server"""
#     if not file:
#         return None
    
#     filename = secure_filename(file.filename)
#     filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    
#     # Save the file
#     file.save(filepath)
    
#     # Optionally, resize the image
#     try:
#         img = Image.open(filepath)
#         img.thumbnail((800, 800))  # Resize to fit within 800x800
#         img.save(filepath)
#     except Exception as e:
#         current_app.logger.error(f"Error processing image: {e}")
    
#     return filename
# def get_image_url(filename):
#     """Get the URL for an image file"""
#     if not filename:
#         return None
    
#     return os.path.join(current_app.config['UPLOAD_FOLDER'], filename)

# def delete_image(filename):
#     """Delete an image file from the server"""
#     if not filename:
#         return
    
#     filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
#     if os.path.exists(filepath):
#         os.remove(filepath)
#         current_app.logger.info(f"Deleted image: {filepath}")
#     else:
#         current_app.logger.warning(f"Image not found: {filepath}")

# def list_images():
#     """List all image files in the upload folder"""
#     images = []
#     for filename in os.listdir(current_app.config['UPLOAD_FOLDER']):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
#             images.append(filename)
#     return images
# def validate_image(file):
#     """Validate if the uploaded file is an image"""
#     if not file:
#         return False
    
#     valid_extensions = ['.png', '.jpg', '.jpeg', '.gif']
#     filename = secure_filename(file.filename)
    
#     if any(filename.lower().endswith(ext) for ext in valid_extensions):
#         return True
    
#     current_app.logger.error(f"Invalid file type: {filename}")
#     return False

# # route to handle image upload
# from flask import Blueprint, request, jsonify
# from app.utils.images import save_image, get_image_url, delete_image, list_images, validate
# _image
# bp = Blueprint('images', __name__)
# @bp.route('/upload', methods=['POST'])
# def upload_image():
#     """Handle image upload"""
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     if not validate_image(file):
#         return jsonify({'error': 'Invalid file type'}), 400
    
#     filename = save_image(file)
    
#     if not filename:
#         return jsonify({'error': 'Failed to save image'}), 500
    
#     image_url = get_image_url(filename)
    
#     return jsonify({'filename': filename, 'url': image_url}), 201
# @bp.route('/images', methods=['GET'])
# def get_images():
#     """List all uploaded images"""
#     images = list_images()
#     return jsonify({'images': images}), 200
# @bp.route('/delete/<filename>', methods=['DELETE'])
# def delete_image_route(filename):
#     """Delete an image by filename"""
#     if not filename:
#         return jsonify({'error': 'Filename is required'}), 400
    
#     delete_image(filename)
    
#     return jsonify({'message': f'Image {filename} deleted successfully'}), 200

# # feed uploaded images to machine learning model
# from app.utils.images import save_image, get_image_url, delete_image, list_images, validate_image
# from app.extensions import db
# from app.models.image import ImageModel  # Assuming you have an ImageModel defined
# def feed_image_to_model(file):
#     """Feed an uploaded image to the machine learning model"""
#     if not validate_image(file):
#         return None
    
#     filename = save_image(file)
    
#     if not filename:
#         return None
    
#     image_url = get_image_url(filename)
    
#     # Assuming you have a function to process the image with your model
#     result = process_image_with_model(image_url)  # Replace with your actual model processing function
    
#     # Save the image metadata to the database
#     new_image = ImageModel(filename=filename, url=image_url, result=result)
#     db.session.add(new_image)
#     db.session.commit()
    
#     return new_image
# def process_image_with_model(image_url):
#     """Process the image with your machine learning model"""
#     # Placeholder for actual model processing logic
#     # This should return the result of the model prediction
#     return "Processed result for " + image_url

# # split uploaded images into training and validation sets
# import os
# import random
# from shutil import copyfile
# def split_images(directory, train_ratio=0.8):
#     """Split images into training and validation sets"""
#     images = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
#     random.shuffle(images)
    
#     split_index = int(len(images) * train_ratio)
#     train_images = images[:split_index]
#     val_images = images[split_index:]
    
#     train_dir = os.path.join(directory, 'train')
#     val_dir = os.path.join(directory, 'val')
    
#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(val_dir, exist_ok=True)
    
#     for img in train_images:
#         copyfile(os.path.join(directory, img), os.path.join(train_dir, img))
    
#     for img in val_images:
#         copyfile(os.path.join(directory, img), os.path.join(val_dir, img))
    
#     return train_images, val_images
# # function to validate image dimensions
# def validate_image_dimensions(file, min_size=(100, 100), max_size=(2000, 2000)):
#     """Validate image dimensions"""
#     try:
#         img = Image.open(file)
#         width, height = img.size
        
#         if (min_size[0] <= width <= max_size[0]) and (min_size[1] <= height <= max_size[1]):
#             return True
#         else:
#             return False
#     except Exception as e:
#         current_app.logger.error(f"Error validating image dimensions: {e}")
#         return False

# # example of usage uploaded images with machine learning model
# from app.utils.images import feed_image_to_model
# def process_uploaded_image(file):
#     """Process an uploaded image and feed it to the model"""
#     if not validate_image(file):
#         return None
    
#     image_data = feed_image_to_model(file)
    
#     if not image_data:
#         return None
    
#     return {
#         'filename': image_data.filename,
#         'url': image_data.url,
#         'result': image_data.result
#     }
# from flask import Blueprint, render_template, request, redirect, url_for
# bp = Blueprint('predict', __name__)
# @bp.route('/predict', methods=['GET', 'POST'])
# def predict():
#     """Prediction home page"""
#     if request.method == 'POST':
#         file = request.files.get('file')
#         if file:
#             result = process_uploaded_image(file)
#             if result:
#                 return render_template('predict/result.html', result=result)
#             else:
#                 return redirect(url_for('predict.predict'))
    
#     return render_template('predict/index.html', title='Predict Home')
# # route to handle image prediction
# @bp.route('/predict/image', methods=['POST'])
# def predict_image():
#     """Handle image prediction"""
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     if not validate_image(file):
#         return jsonify({'error': 'Invalid file type'}), 400
    
#     result = process_uploaded_image(file)
    
#     if not result:
#         return jsonify({'error': 'Failed to process image'}), 500
    
#     return jsonify(result), 200

# # predicting with machine learning model
# from app.utils.images import feed_image_to_model
# def predict_with_model(file):
#     """Predict using the machine learning model with the uploaded image"""
#     if not validate_image(file):
#         return None
    
#     image_data = feed_image_to_model(file)
    
#     if not image_data:
#         return None
    
#     # Assuming the model returns a prediction result
#     prediction_result = image_data.result
    
#     return {
#         'filename': image_data.filename,
#         'url': image_data.url,
#         'prediction': prediction_result
#     }
# from flask import Blueprint, render_template, request, jsonify
# bp = Blueprint('predict', __name__)
# @bp.route('/predict', methods=['GET', 'POST'])
# def predict():
#     """Prediction home page"""
#     if request.method == 'POST':
#         file = request.files.get('file')
#         if file:
#             result = predict_with_model(file)
#             if result:
#                 return render_template('predict/result.html', result=result)
#             else:
#                 return jsonify({'error': 'Failed to process image'}), 500
    
#     return render_template('predict/index.html', title='Predict Home')

# # predicting with machine learning model from server page and displaying results
# from flask import Blueprint, render_template, request, jsonify
# bp = Blueprint('predict', __name__)
# @bp.route('/predict', methods=['GET', 'POST'])
# def predict():
#     """Prediction home page"""
#     if request.method == 'POST':
#         file = request.files.get('file')
#         if file:
#             result = predict_with_model(file)
#             if result:
#                 return render_template('predict/result.html', result=result)
#             else:
#                 return jsonify({'error': 'Failed to process image'}), 500
    
#     return render_template('predict/index.html', title='Predict Home')
# from flask import Blueprint, render_template, request, jsonify
# bp = Blueprint('predict', __name__)
# @bp.route('/predict', methods=['GET', 'POST'])
# def predict():
#     """Prediction home page"""
#     if request.method == 'POST':
#         file = request.files.get('file')
#         if file:
#             result = predict_with_model(file)
#             if result:
#                 return render_template('predict/result.html', result=result)
#             else:
#                 return jsonify({'error': 'Failed to process image'}), 500
    
#     return render_template('predict/index.html', title='Predict Home')
# # run model it self from server page
# from flask import Blueprint, render_template, request, jsonify
# bp = Blueprint('predict', __name__)
# @bp.route('/predict', methods=['GET', 'POST'])
# def predict():
#     """Prediction home page"""
#     if request.method == 'POST':
#         file = request.files.get('file')
#         if file:
#             result = predict_with_model(file)
#             if result:
#                 return render_template('predict/result.html', result=result)
#             else:
#                 return jsonify({'error': 'Failed to process image'}), 500
    
#     return render_template('predict/index.html', title='Predict Home')

# # predicting with machine learning model from server page and displaying results without uploaded images
# from flask import Blueprint, render_template, request, jsonify
# bp = Blueprint('predict', __name__)
# @bp.route('/predict', methods=['GET', 'POST'])
# def predict():
#     """Prediction home page"""
#     if request.method == 'POST':
#         # Assuming you have a way to get the image data without uploading
#         image_data = request.form.get('image_data')
#         if image_data:
#             result = predict_with_model(image_data)
#             if result:
#                 return render_template('predict/result.html', result=result)
#             else:
#                 return jsonify({'error': 'Failed to process image'}), 500
    
#     return render_template('predict/index.html', title='Predict Home')

# # get images from directories and feed it to the model i
# import os
# from app.utils.images import read_csv_from_directory, csv_to_db_bulk
# def process_images_from_directory(directory_path, model):
#     """Process all images from a directory and feed them to the model"""
#     csv_files = read_csv_from_directory(directory_path)
    
#     for csv_file in csv_files:
#         if csv_file.lower().endswith('.csv'):
#             csv_to_db_bulk(csv_file, model)
#             print(f"Processed {csv_file} and fed data to the model.")
    
#     return f"Processed {len(csv_files)} CSV files from {directory_path}."

# # assign images to train or test dataset
# from app.utils.images import split_images
# def assign_images_to_datasets(directory, train_ratio=0.8):
#     """Assign images to training and testing datasets"""
#     train_images, test_images = split_images(directory, train_ratio)
    
#     print(f"Assigned {len(train_images)} images to training set.")
#     print(f"Assigned {len(test_images)} images to testing set.")
    
#     return train_images, test_images



from datetime import datetime
import io
import os
import shutil
from turtle import color
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from app.extensions import db
from app.models.dataset import Dataset
from app.models.dataset_images import DatasetImage
from app.models.traffic_signs import TrafficSign
from app.services.dataset_services import get_test_signs, get_training_signs
from skimage import io, feature, color, transform

def save_image_to_dataset(image_path, class_id, dataset_type, dataset_id):
    """Save image to dataset database and file system"""
    try:
        # Copy file to static/dataset directory
        target_dir = os.path.join('static/dataset', dataset_type, str(class_id))
        os.makedirs(target_dir, exist_ok=True)
        
        filename = os.path.basename(image_path)
        target_path = os.path.join(target_dir, filename)
        shutil.copy2(image_path, target_path)
        
        # Add database entry using passed dataset_id
        relative_path = target_path.replace('static/', '')
        dataset_image = DatasetImage(
            dataset_id=dataset_id,  # ← Use passed dataset_id
            image_path=relative_path,
            class_id=class_id,
            dataset_type=dataset_type
        )
        db.session.add(dataset_image)
        
        return dataset_image
        
    except Exception as e:
        db.session.rollback()
        raise e


def prepare_images(num_classes):
    """Prepare image data for neural network"""
    # Get data
    train_signs = get_training_signs()
    test_signs = get_test_signs()
    
    # Load images
    X_train, y_train = _load_images(train_signs)
    X_test, y_test = _load_images(test_signs)
    
    # Normalize and convert labels
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    return X_train, X_test, y_train, y_test

def _load_images(signs):
    """Load and resize images"""
    images = []
    labels = []
    
    for sign in signs:
        try:
            image = cv2.imread(sign.image_path)
            if image is not None:
                image = cv2.resize(image, (64, 64))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                labels.append(sign.class_id)
        except:
            continue
    
    return np.array(images), np.array(labels)


def load_images_from_directory(dataset_path, load_type):
    # Create or get dataset record
    dataset = Dataset(
        name=f"Traffic Signs {datetime.now().strftime('%Y%m%d_%H%M')}",
        status='loading'
    )
    db.session.add(dataset)
    db.session.flush()  # Get ID
    
    loaded_count = 0
    classes_found = set()
    
    for dataset_type in ['Train', 'Test']:
        type_path = os.path.join(dataset_path, dataset_type)
        if not os.path.exists(type_path):
            continue
            
        for class_dir in os.listdir(type_path):
            class_path = os.path.join(type_path, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            try:
                class_id = int(class_dir)
                if class_id < 0 or class_id > 42:
                    continue
                
                classes_found.add(class_id)  # Track unique classes
                    
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.ppm')):
                        img_path = os.path.join(class_path, img_file)
                        save_image_to_dataset(img_path, class_id, dataset_type.lower(), dataset.id)
                        loaded_count += 1
                        
            except ValueError:
                continue
    
    # ✅ Fix: Update dataset with correct counts
    dataset.status = 'ready'
    dataset.total_images = loaded_count
    dataset.num_classes = len(classes_found)
    
    db.session.commit()
    
    print(f"Final count: {loaded_count} images, {len(classes_found)} classes")  # Debug
    
    return loaded_count

def get_sample_images(dataset_id, samples_per_class=3):
    if not dataset_id:
        return {}
    
    samples = {}
    for class_id in range(43):
        images = DatasetImage.query.filter_by(
            dataset_id=dataset_id, 
            class_id=class_id
        ).limit(samples_per_class).all()
        if images:
            print(f"Class {class_id} images: {[img.image_path for img in images]}")  # Debug
            samples[class_id] = images
    return samples

def get_images_query(class_filter, dataset_type):
    """Get filtered images query"""
    query = DatasetImage.query
    
    if class_filter:
        query = query.filter(DatasetImage.class_id == int(class_filter))
    
    if dataset_type:
        query = query.filter(DatasetImage.dataset_type == dataset_type)
    
    return query

def get_available_classes():
    """Get list of available class IDs"""
    classes = db.session.query(DatasetImage.class_id).distinct().all()
    return sorted([c[0] for c in classes])


def extract_features_from_image(image_path):
    """Extract HOG and color features from image for ML prediction"""
    import numpy as np
    from skimage import io, feature, color, transform
    
    # Load image
    image = io.imread(image_path)
    
    # Handle different image formats
    if len(image.shape) == 3:
        # Color image - convert to grayscale for HOG
        if image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]  # Remove alpha channel
        image_gray = color.rgb2gray(image)
        image_rgb = image  # Keep original for color features
    elif len(image.shape) == 2:
        # Already grayscale
        image_gray = image
        # Create fake RGB for color features
        image_rgb = np.stack([image, image, image], axis=2)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    # Resize to standard size
    image_gray_resized = transform.resize(image_gray, (32, 32))
    image_rgb_resized = transform.resize(image_rgb, (32, 32))
    
    # Extract HOG features from grayscale image
    hog_features = feature.hog(
        image_gray_resized,  # This is guaranteed to be 2D
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )
    
    # Extract color features from RGB image
    if len(image_rgb_resized.shape) == 3 and image_rgb_resized.shape[2] >= 3:
        color_features = [
            np.mean(image_rgb_resized[:, :, 0]),  # Mean Red
            np.mean(image_rgb_resized[:, :, 1]),  # Mean Green
            np.mean(image_rgb_resized[:, :, 2]),  # Mean Blue
            np.std(image_rgb_resized[:, :, 0]),   # Std Red
            np.std(image_rgb_resized[:, :, 1]),   # Std Green
            np.std(image_rgb_resized[:, :, 2])    # Std Blue
        ]
    else:
        color_features = [0] * 6
    
    # Combine features
    features = np.concatenate([hog_features, color_features])
    return features.tolist()