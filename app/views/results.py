import os
from flask_login import current_user
import pandas as pd
from flask import Blueprint, flash, render_template, redirect, request, send_file, url_for

from app.extensions import db
from app.ml.training import get_model_comparison
from app.models.model_result import ModelResult
from app.models.results import Results
from app.models.trained_models import TrainedModel
from app.models.user import User
from app.utils.models import calculate_model_performance


bp = Blueprint('results', __name__)
@bp.route('/')
@bp.route('/results-history')
def results_history():
    """Show prediction results history"""
    page = request.args.get('page', 1, type=int)
    
    # Filter by predicted class if specified
    query = Results.query.filter(Results.profile_id == current_user.id)
    predicted_class = request.args.get('predicted_class')
    if predicted_class:
        query = query.filter(Results.prediction == predicted_class)
    
    results = query.order_by(Results.created_at.desc()).paginate(
        page=page, per_page=10, error_out=False
    )
    
    # Stats
    total_results = Results.query.filter(Results.profile_id == current_user.id).count()
    
    # Fix this - execute the query with .count()
    from datetime import datetime, timedelta
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    recent_results_count = Results.query.filter(
        Results.profile_id == current_user.id,
        Results.created_at >= seven_days_ago
    ).count()  # ✅ Add .count() here
    
    # Get unique predicted classes
    unique_classes = db.session.query(Results.prediction).filter(
        Results.profile_id == current_user.id
    ).distinct().all()
    unique_classes = sorted([c[0] for c in unique_classes])
    
    return render_template('results_history.html',
                         results=results,
                         total_results=total_results,
                         recent_results_count=recent_results_count,  # ✅ Pass count
                         unique_classes=unique_classes)
@bp.route('/model-results')
def model_results():
    """Show REAL model performance results"""
    # Get all active models
    models = TrainedModel.query.filter_by(is_active=True).all()
    
    model_performance = []
    for model in models:
        # Calculate real-time performance
        performance = calculate_model_performance(model.id)
        if performance:
            model_performance.append({
                'model': model,
                'performance': performance
            })
    
    # Also get stored ModelResult records
    stored_results = ModelResult.query.order_by(ModelResult.created_at.desc()).all()
    
    return render_template('model_results.html', 
                         model_performance=model_performance,
                         stored_results=stored_results)

@bp.route('/delete-result/<int:result_id>', methods=['POST'])
def delete_result(result_id):
    """Delete a prediction result"""
    try:
        result = Results.query.get_or_404(result_id)
        db.session.delete(result)
        db.session.commit()
        flash('Result deleted', 'info')
    except Exception as e:
        flash(f'Error deleting result: {str(e)}', 'error')
        db.session.rollback()
    
    return redirect(url_for('predict.results_history'))

@bp.route('/download_results')
def download_results():
    try:
        if os.path.exists('hyper_parameter_results.csv'):
            return send_file('hyper_parameter_results.csv', as_attachment=True)
        else:
            flash('No hyper parameter results file found', 'error')
            return redirect(url_for('results.results'))
    except Exception as e:
        flash(f"An error occurred while downloading results: {str(e)}", 'danger')
        return redirect(url_for('results.results'))