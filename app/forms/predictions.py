import os
from flask_wtf import FlaskForm
from wtforms import FileField, SelectField, SubmitField
from wtforms.validators import DataRequired
from flask_wtf.file import FileRequired, FileAllowed

from app.models.dataset_images import DatasetImage
from app.models.trained_models import TrainedModel
class PredictionForm(FlaskForm):
    model_id = SelectField('Select Model', coerce=int, validators=[DataRequired()])
    
    prediction_type = SelectField('Prediction Type', choices=[
        ('upload', 'Upload New Image'),
        ('uploaded', 'Select from Uploaded Images'),
        ('database', 'Select from Database')
    ], default='upload')
    
    # For file upload
    image = FileField('Upload Image', validators=[
        FileAllowed(['jpg', 'jpeg', 'png', 'bmp', 'ppm'], 'Images only!')
    ])
    
    # For uploaded images selection
    uploaded_image_id = SelectField('Select Uploaded Image', coerce=int)
    
    # For database selection
    database_image_id = SelectField('Select Image from Database', coerce=int)
    
    submit = SubmitField('Predict')
    
    def __init__(self, *args, **kwargs):
        super(PredictionForm, self).__init__(*args, **kwargs)
        
        # Populate model choices
        models = TrainedModel.query.filter_by(is_active=True).all()
        self.model_id.choices = [(0, 'Choose a model...')] + [
            (m.id, f"{m.name} ({m.model_type} - {m.framework})")
            for m in models
        ]
        
        # Populate uploaded images
        uploaded_images = DatasetImage.query.filter_by(dataset_type='uploaded').order_by(DatasetImage.created_at.desc()).limit(50).all()
        self.uploaded_image_id.choices = [(0, 'Choose an uploaded image...')] + [
            (img.id, f"{os.path.basename(img.image_path)} ({img.created_at.strftime('%m/%d %H:%M') if img.created_at else 'Unknown'})")
            for img in uploaded_images
        ]
        
        # Populate database images
        database_images = DatasetImage.query.filter_by(dataset_type='train').limit(100).all()
        self.database_image_id.choices = [(0, 'Choose a database image...')] + [
            (img.id, f"Class {img.class_id} - {os.path.basename(img.image_path)}")
            for img in database_images
        ]