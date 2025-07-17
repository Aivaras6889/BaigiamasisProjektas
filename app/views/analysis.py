from flask import Blueprint, flash, render_template

from app.services.dataset_services import get_test_signs, get_training_signs
from app.utils.dataset import get_dataset_statistics
from app.extensions import db

bp = Blueprint('analysis', __name__)

@bp.route('/')
@bp.route('/dataset_analysis')
def dataset_analysis():
    """Display detailed dataset analysis"""
    try:
        # Get comprehensive dataset statistics
        stats = get_dataset_statistics()
        
        # Get class distribution details
        training_signs = get_training_signs()
        test_signs = get_test_signs()
        
        # Calculate detailed class analysis
        class_analysis = []
        for class_id in range(43):  # Classes 0-42
            train_count = len([s for s in training_signs if s.class_id == class_id])
            test_count = len([s for s in test_signs if s.class_id == class_id])
            total_count = train_count + test_count
            
            class_analysis.append({
                'class_id': class_id,
                'training_count': train_count,
                'test_count': test_count,
                'total_count': total_count,
                'balance_ratio': train_count / test_count if test_count > 0 else 'N/A'
            })
        
        # Sort by total count for better visualization
        class_analysis.sort(key=lambda x: x['total_count'], reverse=True)
        
        db.session.close()
        return render_template('dataset_analysis.html',
                             stats=stats,
                             class_analysis=class_analysis)
    except Exception as e:
        db.session.close()
        flash(f'Error loading dataset analysis: {str(e)}', 'error')
        return render_template('dataset_analysis.html',
                             stats=None,
                             class_analysis=[])
