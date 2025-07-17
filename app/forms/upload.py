
from flask_wtf import FlaskForm
from wtforms.validators import DataRequired
from flask_wtf.file import FileRequired, FileAllowed
from wtforms import FileField, IntegerField, SelectField, SubmitField, ValidationError

class UploadForm(FlaskForm):
    """Form for file upload"""
    file = FileField('File', validators=[
        FileRequired(), 
        FileAllowed(['jpg', 'jpeg', 'png', 'gif', 'bmp', 'ppm'], 'Images only!')
    ])
    class_id = IntegerField('Class ID', validators=[DataRequired()])
    dataset_type = SelectField(
        'Dataset Type', 
        choices=[('train', 'Training'), ('test', 'Testing')],
        default='train'  # Fixed: matches the choice value
    )
    submit = SubmitField('Upload Image')

    def validate_class_id(self, field):
        """Custom validation for class ID"""
        if field.data < 0:
            raise ValidationError('Class ID must be a positive integer.')
    
    def validate_file(self, field):
        """Custom validation for file size (optional)"""
        if field.data:
            # Example: Check file size (5MB limit)
            field.data.seek(0, 2)  # Seek to end
            size = field.data.tell()
            field.data.seek(0)     # Reset to beginning
            
            if size > 5 * 1024 * 1024:  # 5MB
                raise ValidationError('File size must be less than 5MB.')
