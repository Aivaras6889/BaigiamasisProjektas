from app.models.model_result import ModelResult
from app.ml.models_nn import comprehensive_hyperparameter_testing, create_cnn_model
from app.ml.models import train_all_models
from app.services.model_services import get_model_results


def train_traditional_models():
    """Train all traditional ML models"""
    traditional_models = train_all_models()
    return traditional_models

def train_neural_networks():
    """Train neural networks with hyperparameter testing"""
    results_df, best_model = comprehensive_hyperparameter_testing()
    return {
        'best_accuracy': float(results_df['Test_Accuracy'].max()),
        'total_experiments': len(results_df),
        'results_file': 'hyperparameter_results.csv'
    }

def get_model_comparison():
    """Get comparison of all trained models"""
    results = get_model_results()
    
    comparison_data = []
    for result in results:
        comparison_data.append({
            'Model': result.model_name,
            'Type': result.model_type,
            'Accuracy': result.accuracy,
            'Precision': result.precision,
            'Recall': result.recall,
            'F1_Score': result.f1_score,
            'Training_Time': result.training_time
        })
    
    return comparison_data