

from flask import Blueprint, flash, render_template
from app.extensions import db

from app.forms.hyperparameters import HyperparameterForm
from app.ml.models_nn import create_cnn_model, train_with_hyperparameters
from app.utils.images import prepare_images


bp = Blueprint('hyper_parameters', __name__)
@bp.route('/hyper_parameters')
def hyper_parameters():
    form = HyperparameterForm()
    training_results = None  # Placeholder for training results

    if form.validate_on_submit():
        try:
            hyper_params ={
                'model_type': 'cnn',
                'num_conv_layers': form.num_conv_layers.data,
                'conv1_filters': form.conv1_filters.data,
                'conv1_kernel': form.conv1_kernel.data,
                'conv2_filters': form.conv2_filters.data,
                'conv2_kernel': form.conv2_kernel.data,
                'conv3_filters': form.conv3_filters.data,
                'conv3_kernel': 3,
                'conv4_filters':256,
                'conv4_kernel': 3,
                'num_dense_layers': form.num_dense_layers.data,
                'dense1_units': form.dense1_units.data,
                'dense2_units': form.dense2_units.data,
                'dense3_units': 64,
                'dropout1': form.dropout1.data,
                'dropout2': form.dropout2.data,
                'dropout3': 0.5,
                'learning_rate': form.learning_rate.data,
                'batch_size': form.batch_size.data,
                'optimizer': form.optimizer.data,
                'epochs': form.epochs.data    
            }

            X_train, X_test, y_train, y_test = prepare_images()

            result, model = train_with_hyperparameters(X_train,X_test, y_train,y_test, hyper_params)    
            if result:
                model.save('custom_neural_network.h5')
                training_results = result
                flash('Custom model trained successfully with custom hyperparameters.', 'success')
            else:
                flash('Custom model training failed. Please check the parameters.', 'danger')
        except Exception as e:
            flash(f"An error occurred during training: {str(e)}", 'danger')
        finally:
            db.session.close()
    return render_template('hyper_parameters.html', title='Hyperparameters', form=form, result=training_results)

            