from flask import Blueprint, flash, render_template, redirect, url_for
from app.extensions import db
from app.forms.training import TrainingForm
from app.ml.training import train_neural_networks, train_traditional_models

bp= Blueprint('train', __name__)
@bp.route('/')
@bp.route('/train')
def train():
    form  = TrainingForm()
    if form.validate_on_submit():
        try:
            results = {}
            if form.model_type.data in ['traditional', 'both']:
                flash('Training traditional model...', 'info')
                traditional_results = train_traditional_models()
                results['traditional'] = traditional_results
                flash('Traditional model trained successfully.', 'success')
            
            if form.model_type.data in ['neural', 'both']:
                flash('Training neural network model...', 'info')
                neural_results = train_neural_networks()
                results['neural'] = neural_results
                flash('Neural network model trained successfully.', 'success')

            return render_template('train.html', results=results)
        except Exception as e:
            flash(f"An error occurred during training: {str(e)}", 'danger')
        finally:
            db.session.close()
    return render_template('train.html', title='Train Models', form=form, results={})
    
