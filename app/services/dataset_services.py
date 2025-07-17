from app.extensions import db
from app.models.dataset import Dataset
from app.models.traffic_signs import TrafficSign


def get_training_signs():
    signst= TrafficSign.query.filter(TrafficSign.is_training==True).scalar()
    training_signs= db.session.query(signst).all()
    return training_signs

def get_test_signs():
    signs = TrafficSign.query.filter(TrafficSign.is_training==False).scalar()
    test = db.session.query(signs).all()
    return test

def training_signs_count():
    return TrafficSign.query.filter(TrafficSign.is_training==True).count()


def test_signs_count():
    return TrafficSign.query.filter(TrafficSign.is_training==False).count()

def get_current_dataset(dataset_id=None):
    """Get current active dataset"""
    if dataset_id:
        return Dataset.query.get(dataset_id)
    return Dataset.query.order_by(Dataset.created_at.desc()).first()

def get_dataset_history():
    """Get list of all datasets"""
    return Dataset.query.order_by(Dataset.created_at.desc()).all()