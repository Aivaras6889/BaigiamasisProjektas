
from flask_wtf import FlaskForm
from wtforms.validators import DataRequired
from wtforms import SelectField, StringField, SubmitField


class DatasetManagementForm(FlaskForm):
    validate_btn = SubmitField('Validate Dataset')
    export_btn = SubmitField('Export Dataset') 
    delete_btn = SubmitField('Delete Dataset')