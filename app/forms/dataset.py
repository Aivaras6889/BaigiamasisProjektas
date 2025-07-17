from flask_wtf import FlaskForm
from wtforms.validators import DataRequired
from wtforms import BooleanField, SelectField, StringField, SubmitField



class DatasetLoaderForm(FlaskForm):
    dataset_path = StringField('Dataset Directory Path', validators=[DataRequired()])
    load_type = SelectField('Load Type', choices=[
        ('full', 'Load Full Dataset'),
        ('sample', 'Load Sample Only'),
        ('append', 'Append to Existing')
    ], default='full')
    clear_existing = BooleanField('Clear Existing Dataset')
    submit = SubmitField('Load Dataset')
    
