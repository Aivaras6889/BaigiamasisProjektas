from flask import Blueprint, flash, render_template, redirect, request, url_for

from app.utils.visualizations import plot_class_distribution, plot_class_performance_analysis, plot_feature_importance_analysis, plot_hyperparameter_analysis, plot_model_comparison
from app.extensions import db

bp= Blueprint('visualizations', __name__)
@bp.route('/')
@bp.route('/visualizations', methods=['GET', 'POST'])
def visualizations():
    chart_path = None
    chart_type = None
    
    if request.method == 'POST':
        chart_type = request.form.get('chart_type')
        
        try:
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
            
            if chart_path:
                flash(f'Chart generated successfully!', 'success')
            else:
                flash('Chart generation failed!', 'error')
                
        except Exception as e:
            flash(f'Error generating chart: {str(e)}', 'error')
        finally:
            db.session.close()
    
    return render_template('visualizations.html', 
                         chart_path=chart_path, 
                         chart_type=chart_type)