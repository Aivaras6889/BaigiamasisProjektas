

from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from app.forms.batch import BatchUploadForm
from app.forms.csv import CSVImportForm
from app.forms.dataset import DatasetLoaderForm
from app.forms.dataset_management import DatasetManagementForm
from app.services.dataset_services import get_current_dataset, get_dataset_history
from app.utils.dataset import delete_current_dataset, delete_dataset_by_id, export_current_dataset, validate_current_dataset
from app.utils.handlers import handle_directory_loading, handle_csv_import, handle_batch_upload
from app.utils.images import get_available_classes, get_images_query, get_sample_images

bp = Blueprint('dataset', __name__)
@bp.route('/')
@bp.route('/dataset_loader', methods=['GET', 'POST'])
def dataset_loader():
    form = DatasetLoaderForm()
    csv_form = CSVImportForm()
    batch_form = BatchUploadForm()

    if form.validate_on_submit() and form.submit.data:
        handle_directory_loading(form)
        return handle_directory_loading(form)

    if csv_form.validate_on_submit() and csv_form.submit.data:
        return handle_csv_import(csv_form)
        
    
    if batch_form.validate_on_submit() and batch_form.submit.data:
        return handle_batch_upload(batch_form)

    return render_template('dataset/dataset_loader.html', 
                         form=form, 
                         csv_form=csv_form, 
                         batch_form=batch_form)
    

# @bp.route('/dataset/management')

# def dataset_management(dataset_id=None):
#     current_dataset = get_current_dataset(dataset_id)
#     dataset_history = get_dataset_history()
#     validation_results = session.get('validation_results')
#     sample_images = get_sample_images(current_dataset.id if current_dataset else None)
    
#     return render_template('dataset/dataset_management.html',
#                          current_dataset=current_dataset,
#                          dataset_history=dataset_history,
#                          validation_results=validation_results,
#                          sample_images=sample_images)

@bp.route('/management', methods=['GET', 'POST'])
@bp.route('/dataset/management/<int:dataset_id>')
def dataset_management():
    form = DatasetManagementForm()
    current_dataset = get_current_dataset()
    
    if form.validate_on_submit():
        if form.validate_btn.data:
            results = validate_current_dataset()
            session['validation_results'] = results
            flash('Dataset validation completed', 'info')
        elif form.export_btn.data:
            export_path = export_current_dataset()
            flash(f'Dataset exported to: {export_path}', 'success')
        elif form.delete_btn.data:
            delete_current_dataset()
            flash('Dataset deleted successfully', 'success')
            return redirect(url_for('dataset.dataset_loader'))
    
    # GET request or after POST - show template
    return render_template('dataset/dataset_management.html',
                         form=form,
                         current_dataset=current_dataset,
                         dataset_history=get_dataset_history(),
                         validation_results=session.get('validation_results'),
                         sample_images=get_sample_images(current_dataset.id if current_dataset else None))

@bp.route('/dataset/explorer')
def dataset_explorer():
    page = request.args.get('page', 1, type=int)
    class_filter = request.args.get('class_filter', '')
    dataset_type = request.args.get('dataset_type', '')
    per_page = request.args.get('per_page', 20, type=int)
    
    images_query = get_images_query(class_filter, dataset_type)
    pagination = images_query.paginate(page=page, per_page=per_page, error_out=False)
    available_classes = get_available_classes()
    
    return render_template('dataset/dataset_explorer.html',
                         images=pagination.items,
                         total_images=pagination.total,
                         page=page,
                         has_prev=pagination.has_prev,
                         has_next=pagination.has_next,
                         prev_num=pagination.prev_num,
                         next_num=pagination.next_num,
                         available_classes=available_classes,
                         selected_class=class_filter,
                         dataset_type=dataset_type,
                         per_page=per_page)

@bp.route('/delete/<int:dataset_id>', methods=['DELETE', 'POST'])
def delete_dataset(dataset_id):
    try:
        delete_dataset_by_id(dataset_id)
        flash('Dataset deleted successfully', 'success')
        return redirect(url_for('dataset.dataset_management'))
    except Exception as e:
        flash(f'Error deleting dataset: {str(e)}', 'error')
        return redirect(url_for('dataset.dataset_management'))
    