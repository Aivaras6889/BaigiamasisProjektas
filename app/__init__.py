from flask import Flask
from app.extensions import db, migrate, login_manager, csrf
from app.config import CHARTS_FOLDER, UPLOAD_FOLDER, Config
from app.models.user import User

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['CHART_FOLDER'] = CHARTS_FOLDER

    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    csrf.init_app(app)
    
    # Register user loader
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    # Register blueprints
    from app.views.main import bp as main_bp
    app.register_blueprint(main_bp)
    
    from app.views.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')
    
    from app.views.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    from app.views.predict import bp as predict_bp
    app.register_blueprint(predict_bp, url_prefix='/predict')
    
    from app.views.upload import bp as upload_bp
    app.register_blueprint(upload_bp, url_prefix='/upload')
    
    from app.views.train import bp as train_bp
    app.register_blueprint(train_bp, url_prefix='/train')

    from app.views.results import bp as results_bp
    app.register_blueprint(results_bp, url_prefix='/results')

    from app.views.visualizations import bp as visualizations_bp
    app.register_blueprint(visualizations_bp, url_prefix='/visualizations')

    from app.views.hyper_parameters import bp as hyper_parameters_bp
    app.register_blueprint(hyper_parameters_bp, url_prefix='/hyper_parameters')
    
    from app.views.analysis import bp as analysis_bp
    app.register_blueprint(analysis_bp, url_prefix='/analysis')

    from app.views.dataset import bp as dataset_bp
    app.register_blueprint(dataset_bp, url_prefix='/dataset')
    
    from app.views.models import bp as models_bp
    app.register_blueprint(models_bp, url_prefix='/dataset')
    
    return app