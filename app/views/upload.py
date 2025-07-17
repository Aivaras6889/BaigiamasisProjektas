
from flask import Blueprint, render_template, flash, redirect, url_for, request
from app.forms.upload import UploadForm
from app.extensions import db
from app.utils.dataset import get_dataset_statistics
from app.utils.handlers import add_image_to_dataset

# Create a Blueprint for the upload functionality
bp = Blueprint('upload', __name__)
@bp.route('/upload_file', methods=['POST', 'GET'])
def upload_file():
    """Handle file upload"""
    form = UploadForm()
    
    try:
        stats = get_dataset_statistics()
    except:
        stats = None

    if form.validate_on_submit():
        try:
            success, message=add_image_to_dataset(
                form.file.data,
                form.class_id.data,
                form.dataset_type.data == 'train'
            )
            if success:
                flash(message, 'success')
            else:
                flash(message, 'danger')
        except Exception as e:
            flash(f"An error occurred: {str(e)}", 'danger')
        finally:
            db.session.close()
        return redirect(url_for('upload.upload_file'))
    
    return render_template('upload.html', title='Upload File', form=form, stats = stats)
