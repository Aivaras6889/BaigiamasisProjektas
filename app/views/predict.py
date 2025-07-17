from datetime import datetime
import os
from flask import Blueprint, render_template, flash, redirect, session, url_for, request
from flask_login import login_required, current_user
from app.extensions import db
from app.forms.predictions import PredictionForm
from app.ml.predict import get_recent_predictions, predict_single_image, predict_with_model
from app.models.dataset import Dataset
from app.models.dataset_images import DatasetImage
from app.models.results import Results
from app.models.trained_models import TrainedModel
from app.services.predictions import get_predictions_with_offset, total_predictions_count
from werkzeug.utils import secure_filename

from app.utils.images import extract_features_from_image
from app.utils.models import save_directory_model, save_model_performance, save_uploaded_model

bp = Blueprint('predict', __name__)


@bp.route('/')
# @bp.route('/predict', methods=['GET', 'POST'])

# def predict():
#     form = PredictionForm()
#     prediction_result = None
    
#     # Get recent predictions for display
#     try:
#         recent_predictions = get_recent_predictions(10)
#     except:
#         recent_predictions = []
    
#     if form.validate_on_submit():
#         try:
#             result, error = predict_single_image(
#                 form.file.data,
#                 form.model_type.data,
#                 form.model_name.data
#             )
            
#             if error:
#                 flash(error, 'error')
#             else:
#                 prediction_result = result
#                 flash('Prediction completed successfully!', 'success')
#                 # Refresh recent predictions
#                 recent_predictions = get_recent_predictions(10)
                
#         except Exception as e:
#             flash(f'Prediction error: {str(e)}', 'error')
#         finally:
#             db.session.close()
    
#     db.session.close()
#     return render_template('predict.html', 
#                          form=form, 
#                          result=prediction_result,
#                          recent_predictions=recent_predictions)

# @bp.route('/predict', methods=['GET', 'POST'])
@bp.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    """Predict with model and image type selection"""
    form = PredictionForm()
    
    if form.validate_on_submit():
        try:
            model_id = form.model_id.data
            prediction_type = form.prediction_type.data
            
            # Get model info to determine how to handle input
            model = TrainedModel.query.get(model_id)
            if not model:
                flash('Selected model not found', 'error')
                return render_template('predict.html', form=form)
            


            # Handle different prediction types
            if prediction_type == 'upload':
                # Handle new file upload
                if not form.image.data:
                    flash('Please select a file to upload', 'error')
                    return render_template('predict.html', form=form)
                
                file = form.image.data
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
                unique_filename = timestamp + filename
                
               


                upload_dir = 'static/uploads'
                os.makedirs(upload_dir, exist_ok=True)
                filepath = os.path.join(upload_dir, unique_filename)
                file.save(filepath)
                
                # Get or create default dataset
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
                
                # Save to database for future use
                try:
                    uploaded_image = DatasetImage(
                        dataset_id=dataset.id,
                        image_path=f'uploads/{unique_filename}',
                        class_id=-1,
                        dataset_type='uploaded'
                    )
                    db.session.add(uploaded_image)
                    db.session.commit()
                except Exception as db_error:
                    db.session.rollback()
                    print(f"Database error: {db_error}")
                
                # Determine what to pass to prediction function
                if model.framework == 'tensorflow' or model.framework == 'pytorch':
                    # Pass image path for neural networks
                    result = predict_with_model(model_id, filepath)
                else:
                    # Pass HOG features for sklearn/XGBoost/etc
                    features = extract_features_from_image(filepath)
                    result = predict_with_model(model_id, features)
                
                actual_class = None
                image_path = f'uploads/{unique_filename}'
                
            elif prediction_type == 'uploaded':
                # Handle previously uploaded image
                uploaded_image_id = form.uploaded_image_id.data
                if not uploaded_image_id:
                    flash('Please select an uploaded image', 'error')
                    return render_template('predict.html', form=form)
                
                uploaded_image = DatasetImage.query.get_or_404(uploaded_image_id)
                filepath = os.path.join('static', uploaded_image.image_path)
                
                if not os.path.exists(filepath):
                    flash('Image file not found', 'error')
                    return render_template('predict.html', form=form)
                
                # Determine what to pass to prediction function
                if model.framework == 'tensorflow' or model.framework == 'pytorch':
                    result = predict_with_model(model_id, filepath)
                else:
                    features = extract_features_from_image(filepath)
                    result = predict_with_model(model_id, features)
                
                actual_class = None
                image_path = uploaded_image.image_path
                
            else:  # database
                # Handle database selection
                database_image_id = form.database_image_id.data
                if not database_image_id:
                    flash('Please select an image from database', 'error')
                    return render_template('predict.html', form=form)
                
                db_image = DatasetImage.query.get_or_404(database_image_id)
                filepath = os.path.join('static', db_image.image_path)
                
                if not os.path.exists(filepath):
                    flash('Image file not found', 'error')
                    return render_template('predict.html', form=form)
                
                # Determine what to pass to prediction function
                if model.framework == 'tensorflow' or model.framework == 'pytorch':
                    result = predict_with_model(model_id, filepath)
                else:
                    features = extract_features_from_image(filepath)
                    result = predict_with_model(model_id, features)
                
                actual_class = db_image.class_id  # We know the actual class!
                image_path = db_image.image_path
            
            # Add additional info to result
            result['actual_class'] = actual_class
            result['image_path'] = image_path

            try:
                new_result = Results(
                    profile_id=current_user.id,
                    prediction=str(result['prediction']),
                    confidence=result.get('confidence'),
                    actual_class=actual_class,
                    model_name=model.name,  # ✅ Make sure to save model name
                    image_path=image_path,
                    prediction_time=result.get('prediction_time', 0)
                )
                db.session.add(new_result)
                db.session.commit()
                
                # ✅ Update model performance if we have actual class
                if actual_class is not None:
                    save_model_performance(model_id)
                    
            except Exception as save_error:
                print(f"Error saving result: {save_error}")
                db.session.rollback()
            
            flash('Prediction completed successfully!', 'success')
            return render_template('predict.html', form=form, result=result)
            
        except Exception as e:
            flash(f'Prediction error: {str(e)}', 'error')
            db.session.rollback()
    
    return render_template('predict.html', form=form)

@bp.route('/predict-image', methods=['POST'])
def predict_image():
    """Make prediction on uploaded image"""
    form = ""
    try:
        # Get model from session
        model_id = session.get('selected_model_id')
        if not model_id:
            flash('Please load a model first', 'error')
            return redirect(url_for('predict.predict'))
        
        # Handle file upload
        if 'image_file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(url_for('predict.predict'))
        
        file = request.files['image_file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('predict.predict'))
        
        # Save uploaded image
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        unique_filename = timestamp + filename
        
        upload_dir = 'static/uploads'
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, unique_filename)
        file.save(filepath)
        
        # Extract features
        features = extract_features_from_image(filepath)
        
        # Make prediction
        result = predict_with_model(model_id, features)
        
        flash('Prediction completed successfully!', 'success')
        return render_template('predict.html', 
                             form=form, 
                             result=result,
                             image_path=f'uploads/{unique_filename}')
        
    except Exception as e:
        flash(f'Prediction error: {str(e)}', 'error')
        return redirect(url_for('predict.predict'))


# @bp.route('/history')
# def prediction_history():
#     """Simple prediction history"""
#     page = request.args.get('page', 1, type=int)
    
#     # Simple filters
#     query = PredictionHistory.query
    
#     model_id = request.args.get('model_id')
#     if model_id:
#         query = query.filter(PredictionHistory.model_id == model_id)
    
#     result_filter = request.args.get('result')
#     if result_filter == 'correct':
#         query = query.filter(PredictionHistory.is_correct == True)
#     elif result_filter == 'incorrect':
#         query = query.filter(PredictionHistory.is_correct == False)
    
#     predictions = query.order_by(PredictionHistory.created_at.desc()).paginate(
#         page=page, per_page=10, error_out=False
#     )
    
#     # Simple stats
#     total = PredictionHistory.query.count()
#     correct = PredictionHistory.query.filter_by(is_correct=True).count()
#     accuracy = (correct / total * 100) if total > 0 else 0
#     avg_time = db.session.query(db.func.avg(PredictionHistory.prediction_time)).scalar() or 0
    
#     most_used = db.session.query(TrainedModel.name).join(PredictionHistory).group_by(TrainedModel.id).order_by(db.func.count(PredictionHistory.id).desc()).first()
    
#     stats = {
#         'total_predictions': total,
#         'accuracy': accuracy,
#         'avg_time': avg_time,
#         'most_used_model': most_used.name if most_used else 'None'
#     }
    
#     available_models = TrainedModel.query.filter_by(is_active=True).all()
    
#     return render_template('prediction_history.html', 
#                          predictions=predictions, 
#                          stats=stats,
#                          available_models=available_models)