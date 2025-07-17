from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from app.extensions import db
from app.ml.predict import predict_single_image
from app.ml.training import train_neural_networks, train_traditional_models
from app.models import User
from app.utils.dataset import get_dataset_statistics
from app.utils.decorators import json_required
from app.utils.visualizations import plot_class_distribution, plot_class_performance_analysis, plot_feature_importance_analysis, plot_hyperparameter_analysis, plot_model_comparison

bp = Blueprint('api', __name__)


@bp.route('/users/<int:user_id>')
@login_required
def get_user(user_id):
    """Get user information"""
    user = User.query.get_or_404(user_id)
    return jsonify({
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'full_name': user.full_name,
        'created_at': user.created_at.isoformat()
    })


@bp.route('/profile', methods=['PUT'])
@login_required
@json_required
def update_profile():
    """Update user profile via API"""
    data = request.get_json()
    
    if 'first_name' in data:
        current_user.first_name = data['first_name']
    if 'last_name' in data:
        current_user.last_name = data['last_name']
    
    from app.extensions import db
    db.session.commit()
    
    return jsonify({
        'message': 'Profile updated successfully',
        'user': {
            'id': current_user.id,
            'full_name': current_user.full_name
        }
    })
@bp.route('/generate_chart/<chart_type>', methods=['POST'])
@login_required
@json_required
def generate_chart(chart_type):
    if chart_type == 'class_distribution':
        chart_path = plot_class_distribution()
    elif chart_type == 'model_comparison':
        chart_path = plot_model_comparison()
    elif chart_type == 'hyperparameter_analysis':
        chart_path = plot_hyperparameter_analysis()
    elif chart_type == 'class_analysis':
        chart_path = plot_class_performance_analysis()
    elif chart_type == 'feature_analysis':
        chart_path = plot_feature_importance_analysis()
    else:
        return jsonify({'error': 'Invalid chart type'})
    db.session.close()
    if chart_path:
        return jsonify({'success': True, 'chart_path': chart_path})
    else:
        return jsonify({'success': False, 'error': 'Failed to generate chart'})
    
@bp.route('/dataset_stats', methods=['POST'])
@login_required
@json_required
def api_dataset_stats():
    try:
        stats = get_dataset_statistics()
        db.session.close()
        return jsonify(stats)
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@bp.route('/api/predict_image', methods=['POST'])
@login_required
@json_required
def predict_image():
    from flask import request, jsonify
    
    file = request.files.get('file')
    model_type = request.form.get('model_type', 'neural')
    model_name = request.form.get('model_name', 'best_neural_network')
    
    result, error = predict_single_image(file, model_type, model_name)
    
    if error:
        return jsonify({'success': False, 'error': error})
    
    return jsonify({
        'success': True,
        'predicted_class': result['predicted_class'],
        'confidence': result.get('confidence'),
        'model_used': result['model_used']
    })

@bp.route('/api/train_models', methods=['POST'])
@login_required
@json_required
def train_models():

    
    model_type = request.form.get('model_type', 'both')
    
    results = {}
    
    if model_type in ['traditional', 'both']:
        results['traditional'] = train_traditional_models()
    
    if model_type in ['neural', 'both']:
        results['neural'] = train_neural_networks()
    
    return jsonify({
        'success': True,
        'results': results
    })
@bp.route('/api/dataset_stats')
@login_required
@json_required
def get_dataset_stats():
    return jsonify(get_dataset_statistics())

@bp.route('/api/recent_predictions')
@login_required
@json_required
def get_recent_predictions():
    return jsonify(get_recent_predictions())