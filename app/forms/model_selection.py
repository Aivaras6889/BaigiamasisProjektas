
from app.models.trained_models import TrainedModel
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField, SelectField, SubmitField
from wtforms.validators import DataRequired
from app.models import TrainedModel

class UploadModelForm(FlaskForm):
    model_file = FileField('Model File', validators=[
        FileRequired(),
        FileAllowed(['joblib', 'pkl', 'keras', 'h5', 'pt', 'pth'], 'Model files only!')
    ])
    
    model_name = StringField('Model Name', validators=[DataRequired()])
    
    model_type = SelectField('Model Type', choices=[
        ('', 'Choose type...'),
        ('svm', 'Support Vector Machine'),
        ('random_forest', 'Random Forest'),
        ('neural_network', 'Neural Network'),
        ('cnn', 'Convolutional Neural Network'),
        ('xgboost', 'XGBoost'),
        ('lightgbm', 'LightGBM'),
        ('catboost', 'CatBoost'),
        ('logistic_regression', 'Logistic Regression'),
        ('naive_bayes', 'Naive Bayes'),
        ('knn', 'K-Nearest Neighbors')
    ], validators=[DataRequired()])
    
    framework = SelectField('Framework', choices=[
        ('', 'Choose framework...'),
        ('sklearn', 'Scikit-learn'),
        ('tensorflow', 'TensorFlow'),
        ('pytorch', 'PyTorch'),
        ('xgboost', 'XGBoost'),
        ('lightgbm', 'LightGBM'),
        ('catboost', 'CatBoost')
    ], validators=[DataRequired()])
    
    submit = SubmitField('Upload Model')

class SelectModelForm(FlaskForm):
    model_id = SelectField('Available Models', coerce=int, validators=[DataRequired()])
    submit = SubmitField('Load Selected Model')
    
    def __init__(self, *args, **kwargs):
        super(SelectModelForm, self).__init__(*args, **kwargs)
        
        # Populate model choices
        models = TrainedModel.query.filter_by(is_active=True).all()
        self.model_id.choices = [(0, 'Choose a model...')] + [
            (m.id, f"{m.name} ({m.model_type} - {m.framework})")
            for m in models
        ]