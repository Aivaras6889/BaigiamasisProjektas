from app.models.model_result import ModelResult
from app.models.trained_models import TrainedModel


def get_model_results():
    return ModelResult.query.all()

def get_database_models():
    """Get all available models from database"""
    return TrainedModel.query.filter_by(is_active=True).all()