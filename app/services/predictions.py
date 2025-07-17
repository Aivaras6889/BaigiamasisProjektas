from app.extensions import db
from app.models.predictions import Prediction

def get_predictions(limit):
    return Prediction.query.order_by(Prediction.created_at.desc()).limit(limit).all()



"""

        
        predictions = session.query(Prediction).order_by(
            Prediction.created_at.desc()
        ).offset((page-1)*per_page).limit(per_page).all()
        
        total_predictions = session.query(Prediction).count()
        

"""

def get_predictions_with_offset(page,per_page):
    return Prediction.query.order_by(Prediction.created_at.desc()).offset((page-1)*per_page).limit(per_page).all()

def total_predictions_count():
    return Prediction.query.count()
