from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed,FileRequired
from wtforms import FileField, SubmitField

class BatchUploadForm(FlaskForm):
    zip_file = FileField('ZIP File', validators=[
        FileRequired(),
        FileAllowed(['zip'], 'ZIP files only!')
    ])
    submit_batch = SubmitField('Upload Batch')