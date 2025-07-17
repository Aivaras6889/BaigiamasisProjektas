from datetime import datetime
import os
from flask import Blueprint, render_template, flash, redirect, session, url_for, request
from flask_login import login_required
from app.extensions import db
from app.forms.model_selection import SelectModelForm, UploadModelForm
from app.models.trained_models import TrainedModel
from app.services.model_services import get_database_models
from app.utils.models import detect_model_type, get_available_models, load_model_util, save_directory_model, save_uploaded_model
from werkzeug.utils import secure_filename


bp= Blueprint('models', __name__)

@bp.route('/models')
def models_main():
    """Main models page - upload and view models"""
    models = TrainedModel.query.filter_by(is_active=True).order_by(TrainedModel.created_at.desc()).all()
    active_models = len(models)
    
    # Get unique frameworks and model types for stats
    frameworks = list(set([m.framework for m in models if m.framework]))
    model_types = list(set([m.model_type for m in models if m.model_type]))
    
    return render_template('models_main.html', 
                         models=models,
                         active_models=active_models,
                         frameworks=frameworks,
                         model_types=model_types)




@bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_form():
    """Show upload form and handle submission"""
    form = UploadModelForm()
    
    if form.validate_on_submit():
        try:
            # Handle file upload
            file = form.model_file.data
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            unique_filename = timestamp + filename
            
            models_dir = 'static/models'
            os.makedirs(models_dir, exist_ok=True)
            file_path = os.path.join(models_dir, unique_filename)
            file.save(file_path)
            
            # Save to database
            new_model = TrainedModel(
                name=form.model_name.data,
                model_type=form.model_type.data,
                framework=form.framework.data,
                file_path=file_path,
                is_active=True
            )
            
            db.session.add(new_model)
            db.session.commit()
            
            flash(f'Model "{form.model_name.data}" uploaded successfully!', 'success')
            return redirect(url_for('models.models_main'))
            
        except Exception as e:
            flash(f'Error uploading model: {str(e)}', 'error')
            db.session.rollback()
    
    return render_template('upload_model.html', form=form)

@bp.route('/deactivate/<int:model_id>', methods=['POST'])
def deactivate_model(model_id):
    """Deactivate a model"""
    try:
        model = TrainedModel.query.get_or_404(model_id)
        model.is_active = False
        db.session.commit()
        
        flash(f'Model "{model.name}" deactivated', 'info')
    except Exception as e:
        flash(f'Error deactivating model: {str(e)}', 'error')
        db.session.rollback()
    
    return redirect(url_for('models.models_main'))

@bp.route('/activate/<int:model_id>', methods=['POST'])
def activate_model(model_id):
    """Activate a model"""
    try:
        model = TrainedModel.query.get_or_404(model_id)
        model.is_active = True
        db.session.commit()
        
        flash(f'Model "{model.name}" activated', 'success')
    except Exception as e:
        flash(f'Error activating model: {str(e)}', 'error')
        db.session.rollback()
    
    return redirect(url_for('models.models_main'))

@bp.route('/delete/<int:model_id>', methods=['POST'])
def delete_model(model_id):
    """Delete a model permanently"""
    try:
        model = TrainedModel.query.get_or_404(model_id)
        
        # Delete physical file
        if os.path.exists(model.file_path):
            os.remove(model.file_path)
        
        # Delete from database
        db.session.delete(model)
        db.session.commit()
        
        flash(f'Model "{model.name}" deleted permanently', 'warning')
    except Exception as e:
        flash(f'Error deleting model: {str(e)}', 'error')
        db.session.rollback()
    
    return redirect(url_for('models.models_main'))


# @bp.route('/trained', methods=['GET', 'POST'])
# def trained_models():
#     directory_models = []
    
#     # Handle browse directory
#     if request.method == 'POST' and 'browse_directory' in request.form:
#         directory_path = request.form.get('model_directory')
#         print(f"Directory path: {directory_path}")
        
#         if directory_path and os.path.exists(directory_path):
#             print(f"Directory exists, scanning...")
            
#             for filename in os.listdir(directory_path):
#                 print(f"Found file: {filename}")
                
#                 if filename.endswith(('.joblib', '.pkl', '.keras', '.h5', '.pt', '.pth')):
#                     print(f"Valid model file: {filename}")
                    
#                     file_path = os.path.join(directory_path, filename)
                    
#                     # Simple detection instead of function
#                     if 'svm' in filename.lower():
#                         model_type = 'svm'
#                     elif 'rf' in filename.lower() or 'forest' in filename.lower():
#                         model_type = 'random_forest'
#                     elif 'xgb' in filename.lower():
#                         model_type = 'xgboost'
#                     else:
#                         model_type = 'unknown'
                    
#                     print(f"Detected type: {model_type}")
                    
#                     directory_models.append({
#                         'name': filename,
#                         'path': file_path,
#                         'model_type': model_type
#                     })
            
#             print(f"Total models found: {len(directory_models)}")
#             flash(f'Found {len(directory_models)} model files', 'success')
#         else:
#             print("Directory doesn't exist")
#             flash('Directory not found', 'error')
    
#     form = CombinedModelLoaderForm()
#     models = get_database_models()
    
#     return render_template('trained_models.html', 
#                          form=form, 
#                          models=models,
#                          directory_models=directory_models)

# @bp.route('/load-any-model', methods=['POST'])
# def load_any_model():
#     """Handle loading models from any source"""
#     form = CombinedModelLoaderForm()
    
#     if form.validate_on_submit():
#         load_type = form.load_type.data
        
#         try:
#             if load_type == 'database':
#                 # Load from database
#                 model_id = form.database_model_id.data
#                 if not model_id:
#                     flash('Please select a database model', 'error')
#                     return redirect(url_for('models.trained_models'))
                
#                 # Load the selected model
#                 session['selected_model_id'] = model_id
#                 selected_model = TrainedModel.query.get(model_id)
#                 session['selected_model_name'] = selected_model.name
                
#                 flash(f'Model "{selected_model.name}" loaded successfully!', 'success')
                
#             elif load_type == 'file':
#                 # Handle file upload
#                 if not form.model_file.data:
#                     flash('Please select a model file', 'error')
#                     return redirect(url_for('models.trained_models'))
                
#                 model_id = save_uploaded_model(form)
#                 session['selected_model_id'] = model_id
#                 session['selected_model_name'] = form.model_name.data
                
#                 flash(f'File model "{form.model_name.data}" loaded!', 'success')
                
#             elif load_type == 'directory':
#                 # Handle directory model
#                 directory_model = request.form.get('directory_model')
#                 if not directory_model:
#                     flash('Please select a model from directory', 'error')
#                     return redirect(url_for('models.trained_models'))
                
#                 model_id = save_directory_model(directory_model, form)
#                 session['selected_model_id'] = model_id
#                 session['selected_model_name'] = form.model_name.data
                
#                 flash(f'Directory model "{form.model_name.data}" loaded!', 'success')
            
#             else:
#                 # Debug validation errors
#                 print("Form validation failed!")
#                 print("Form errors:", form.errors)
#                 for field, errors in form.errors.items():
#                     for error in errors:
#                         print(f"Field '{field}': {error}")
#                         flash(f"Error in {field}: {error}", 'error')
            
#             return redirect(url_for('models.trained_models'))
            
#         except Exception as e:
#             flash(f'Error loading model: {str(e)}', 'error')
#     else:
#         flash('Form validation failed', 'error')
    
#     return redirect(url_for('models.trained_models'))


# @bp.route('/browse-directory', methods=['POST'])
# def browse_directory():
#     """Browse directory for model files"""
#     form = CombinedModelLoaderForm()
#     directory_models = []
    
#     if form.model_directory.data:
#         directory_path = form.model_directory.data
        
#         if os.path.exists(directory_path):
#             for filename in os.listdir(directory_path):
#                 if filename.endswith(('.joblib', '.pkl', '.keras', '.h5', '.pt', '.pth')):
#                     file_path = os.path.join(directory_path, filename)
#                     model_type = detect_model_type(filename)
                    
#                     directory_models.append({
#                         'name': filename,
#                         'path': file_path,
#                         'type': model_type
#                     })
#         else:
#             flash('Directory not found', 'error')
    
#     return render_template('trained_models.html', 
#                          form=form, 
#                          models=get_database_models(),
#                          directory_models=directory_models)



    