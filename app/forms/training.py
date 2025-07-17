from flask_wtf import FlaskForm
from wtforms import BooleanField, FileField, SelectField, SubmitField
from wtforms.validators import DataRequired
from flask_wtf.file import FileRequired, FileAllowed

class TrainingForm(FlaskForm):
    model_type = SelectField('Model Type to Train', choices=[
        ('traditional', 'Traditional ML Only'),
        ('neural', 'Neural Networks Only'),
        ('both', 'Both Types')
    ], default='both')
    hyperparameter_tuning = BooleanField('Enable Hyperparameter Tuning')
    submit = SubmitField('Start Training')