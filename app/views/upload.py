
from datetime import datetime
import os
from flask import Blueprint, current_app, render_template, flash, redirect, url_for, request
from app.forms.upload import UploadForm
from app.extensions import db
from app.models.dataset import Dataset
from app.models.dataset_images import DatasetImage
from app.utils.dataset import get_dataset_statistics
from app.utils.handlers import add_image_to_dataset
from werkzeug.utils import secure_filename
# Create a Blueprint for the upload functionality
bp = Blueprint('upload', __name__)
@bp.route('/upload_file', methods=['POST', 'GET'])

def upload_file():
    """Handle file upload"""
    form = UploadForm()
    
    # Get dataset statistics
    try:
        stats = get_dataset_statistics()
    except Exception as e:
        current_app.logger.error(f"Error getting dataset statistics: {str(e)}")
        stats = None
    
    if form.validate_on_submit():
        try:
            # Get the uploaded file
            file = form.file.data
            if not file or not file.filename:
                flash('No file selected', 'danger')
                return redirect(url_for('upload.upload_file'))
            
            # Create secure filename with timestamp
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            unique_filename = timestamp + filename
            
            # Use app/static/uploads directory
            upload_folder = os.path.join(current_app.static_folder, 'uploads')
            
            # Check if directory exists
            if not os.path.exists(upload_folder):
                flash('Upload directory not found. Please contact administrator.', 'danger')
                return redirect(url_for('upload.upload_file'))
            
            # Full path for saving the file
            file_path = os.path.join(upload_folder, unique_filename)
            
            # Get or create dataset
            dataset = Dataset.query.first()
            if not dataset:
                dataset = Dataset(
                    name="Default Dataset",
                    status='ready',
                    total_images=0,
                    num_classes=0
                )
                db.session.add(dataset)
                db.session.flush()
            
            # Save file to filesystem
            file.save(file_path)
            
            # Create database record
            uploaded_image = DatasetImage(
                dataset_id=dataset.id,
                image_path=f'uploads/{unique_filename}',
                class_id=form.class_id.data,
                dataset_type=form.dataset_type.data
            )
            
            db.session.add(uploaded_image)
            
            # Update dataset statistics
            dataset.total_images += 1
            existing_classes = db.session.query(DatasetImage.class_id).distinct().count()
            dataset.num_classes = existing_classes
            
            db.session.commit()
            
            flash(f'File "{filename}" uploaded successfully!', 'success')
            return redirect(url_for('upload.upload_file'))
            
        except Exception as e:
            db.session.rollback()
            
            # Clean up file if it was saved but database failed
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            
            current_app.logger.error(f"Upload error: {str(e)}")
            flash(f"An error occurred during upload: {str(e)}", 'danger')
            return redirect(url_for('upload.upload_file'))
        
        finally:
            db.session.close()
    
    # Handle form validation errors
    elif request.method == 'POST':
        for field, errors in form.errors.items():
            for error in errors:
                flash(f"Error in {field}: {error}", 'danger')
    
    return render_template('upload.html', title='Upload File', form=form, stats=stats)


