
from flask_wtf import FlaskForm
from wtforms.validators import DataRequired
from flask_wtf.file import FileRequired, FileAllowed
from wtforms import FileField, IntegerField, SelectField, SubmitField


class UploadForm(FlaskForm):
    """Form for file upload"""
    file = FileField('File', validators=[FileRequired(), FileAllowed(['jpg', 'jpeg', 'png', 'gif', 'bmp', 'ppm'])])
    class_id = IntegerField('Class ID', validators=[DataRequired()])
    dataset_type = SelectField('Dataset Type', choices=[('train', 'Training'), ('test', 'Testing')],default='training')
 
    submit = SubmitField('Upload Image')


    def validate_file(self, field):
        """Custom validation for file type"""
        if not field.data.filename.endswith(('.png', '.jpg', '.jpeg')):
            raise ValueError('Only image files are allowed.')
        return True

