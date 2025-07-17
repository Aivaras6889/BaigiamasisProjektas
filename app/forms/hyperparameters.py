
from flask_wtf import FlaskForm
from wtforms import SelectField, FloatField, IntegerField, SubmitField
from wtforms.validators import NumberRange, DataRequired


class HyperparameterForm(FlaskForm):
    # Conv layer parameters
    num_conv_layers = SelectField('Number of Conv Layers', choices=[
        (2, '2 Layers'), (3, '3 Layers'), (4, '4 Layers')
    ], coerce=int, default=3)
    conv1_filters = SelectField('Conv1 Filters', choices=[
        (16, '16'), (32, '32'), (64, '64')
    ], coerce=int, default=32)
    conv1_kernel = SelectField('Conv1 Kernel Size', choices=[
        (3, '3x3'), (5, '5x5')
    ], coerce=int, default=3)
    conv2_filters = SelectField('Conv2 Filters', choices=[
        (32, '32'), (64, '64'), (128, '128')
    ], coerce=int, default=64)
    conv2_kernel = SelectField('Conv2 Kernel Size', choices=[
        (3, '3x3'), (5, '5x5')
    ], coerce=int, default=3)
    conv3_filters = SelectField('Conv3 Filters', choices=[
        (64, '64'), (128, '128'), (256, '256')
    ], coerce=int, default=128)
    
    # Dense layer parameters
    num_dense_layers = SelectField('Number of Dense Layers', choices=[
        (1, '1 Layer'), (2, '2 Layers'), (3, '3 Layers')
    ], coerce=int, default=2)
    dense1_units = SelectField('Dense1 Units', choices=[
        (64, '64'), (128, '128'), (256, '256'), (512, '512')
    ], coerce=int, default=256)
    dense2_units = SelectField('Dense2 Units', choices=[
        (32, '32'), (64, '64'), (128, '128'), (256, '256')
    ], coerce=int, default=128)
    
    # Dropout
    dropout1 = FloatField('Dropout 1', validators=[
        NumberRange(min=0.0, max=0.9)
    ], default=0.5)
    dropout2 = FloatField('Dropout 2', validators=[
        NumberRange(min=0.0, max=0.9)
    ], default=0.5)
    
    # Training parameters
    learning_rate = SelectField('Learning Rate', choices=[
        (0.1, '0.1'), (0.01, '0.01'), (0.001, '0.001'), (0.0001, '0.0001')
    ], coerce=float, default=0.001)
    batch_size = SelectField('Batch Size', choices=[
        (16, '16'), (32, '32'), (64, '64'), (128, '128')
    ], coerce=int, default=32)
    optimizer = SelectField('Optimizer', choices=[
        ('adam', 'Adam'), ('sgd', 'SGD'), ('rmsprop', 'RMSprop')
    ], default='adam')
    epochs = IntegerField('Epochs', validators=[
        NumberRange(min=1, max=100)
    ], default=20)
    
    submit = SubmitField('Train with Custom Parameters')