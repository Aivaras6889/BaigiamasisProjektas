from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired,FileAllowed
from wtforms import FileField, StringField, SubmitField
from wtforms.validators import Optional


class CSVImportForm(FlaskForm):
    csv_file = FileField('CSV File', validators=[
        FileRequired(),
        FileAllowed(['csv'], 'CSV files only!')
    ])
    base_path = StringField('Base Image Path', validators=[Optional()])
    submit_csv = SubmitField('Import from CSV')