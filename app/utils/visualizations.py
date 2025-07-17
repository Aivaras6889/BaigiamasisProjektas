from typing import Counter
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
from app.extensions import db
from app.config import Config
from app.models.model_result import ModelResult
from app.models.traffic_signs import TrafficSign
from app.utils.features import deserialize_features

def plot_class_distribution(self):
    """Chart 1: Distribution of traffic sign classes"""
    # Get data
    train_signs = self.session.query(TrafficSign).filter(TrafficSign.is_training == True).all()
    test_signs = self.session.query(TrafficSign).filter(TrafficSign.is_training == False).all()
    
    train_classes = [sign.class_id for sign in train_signs]
    test_classes = [sign.class_id for sign in test_signs]
    
    train_counts = Counter(train_classes)
    test_counts = Counter(test_classes)
    
    # Get all classes (0-42)
    all_classes = list(range(43))
    train_values = [train_counts.get(cls, 0) for cls in all_classes]
    test_values = [test_counts.get(cls, 0) for cls in all_classes]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Training distribution
    bars1 = ax1.bar(all_classes, train_values, alpha=0.7, color='skyblue')
    ax1.set_title('Training Data - Class Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Traffic Sign Class (0-42)')
    ax1.set_ylabel('Number of Images')
    ax1.grid(True, alpha=0.3)
    
    # Test distribution
    bars2 = ax2.bar(all_classes, test_values, alpha=0.7, color='lightcoral')
    ax2.set_title('Test Data - Class Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Traffic Sign Class (0-42)')
    ax2.set_ylabel('Number of Images')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(self.output_dir, 'class_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def plot_model_comparison(self):
    """Chart 2: Model performance comparison"""
    # Get model results
    results = self.session.query(ModelResult).all()
    
    if not results:
        return None
    
    # Prepare data
    models = [r.model_name for r in results]
    accuracies = [r.accuracy for r in results]
    model_types = [r.model_type for r in results]
    f1_scores = [r.f1_score for r in results]
    training_times = [r.training_time for r in results]
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Accuracy comparison
    colors = ['lightblue' if t == 'traditional' else 'lightgreen' for t in model_types]
    bars = ax1.bar(range(len(models)), accuracies, color=colors)
    ax1.set_title('Model Accuracy Comparison', fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    
    # Add value labels
    for i, bar in enumerate(bars):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{accuracies[i]:.3f}', ha='center', va='bottom')
    
    # 2. F1 Score comparison
    bars = ax2.bar(range(len(models)), f1_scores, color=colors)
    ax2.set_title('Model F1 Score Comparison', fontweight='bold')
    ax2.set_ylabel('F1 Score')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    
    # 3. Training time comparison
    bars = ax3.bar(range(len(models)), training_times, color=colors)
    ax3.set_title('Training Time Comparison', fontweight='bold')
    ax3.set_ylabel('Training Time (seconds)')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(models, rotation=45, ha='right')
    
    # 4. Accuracy vs Training Time scatter
    ax4.scatter([t for t, mt in zip(training_times, model_types) if mt == 'traditional'],
                [a for a, mt in zip(accuracies, model_types) if mt == 'traditional'],
                color='blue', label='Traditional ML', s=60)
    ax4.scatter([t for t, mt in zip(training_times, model_types) if mt == 'neural'],
                [a for a, mt in zip(accuracies, model_types) if mt == 'neural'],
                color='red', label='Neural Network', s=60)
    ax4.set_xlabel('Training Time (seconds)')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Accuracy vs Training Time')
    ax4.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(self.output_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def plot_hyperparameter_analysis(self):
    """Chart 3: Neural network hyperparameter analysis"""
    try:
        # Load hyperparameter results
        df = pd.read_csv('hyperparameter_results.csv')
        
        if df.empty:
            return None
        
        # Create analysis plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Learning rate vs Accuracy
        if 'learning_rate' in df.columns:
            grouped = df.groupby('learning_rate')['Test_Accuracy'].mean()
            ax1.bar(grouped.index.astype(str), grouped.values, color='lightblue')
            ax1.set_title('Average Accuracy vs Learning Rate')
            ax1.set_xlabel('Learning Rate')
            ax1.set_ylabel('Test Accuracy')
            
            for i, v in enumerate(grouped.values):
                ax1.text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        # 2. Batch size vs Accuracy
        if 'batch_size' in df.columns:
            grouped = df.groupby('batch_size')['Test_Accuracy'].mean()
            ax2.bar(grouped.index.astype(str), grouped.values, color='lightgreen')
            ax2.set_title('Average Accuracy vs Batch Size')
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Test Accuracy')
        
        # 3. Training time vs Accuracy
        if 'Training_Time' in df.columns:
            ax3.scatter(df['Training_Time'], df['Test_Accuracy'], alpha=0.6)
            ax3.set_xlabel('Training Time (seconds)')
            ax3.set_ylabel('Test Accuracy')
            ax3.set_title('Training Time vs Accuracy')
        
        # 4. Optimizer comparison
        if 'optimizer' in df.columns:
            grouped = df.groupby('optimizer')['Test_Accuracy'].mean()
            ax4.bar(grouped.index, grouped.values, color='orange')
            ax4.set_title('Average Accuracy by Optimizer')
            ax4.set_xlabel('Optimizer')
            ax4.set_ylabel('Test Accuracy')
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'hyperparameter_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    except Exception as e:
        print(f"Error creating hyperparameter analysis: {e}")
        return None

def plot_class_performance_analysis():
    """Chart 4: Performance analysis by traffic sign class"""
    # This requires actual predictions - simplified version
    train_signs = db.session.query(TrafficSign).filter(TrafficSign.is_training == True).all()
    
    # Analyze class distribution and difficulty
    class_counts = Counter([sign.class_id for sign in train_signs])
    
    # Create difficulty analysis based on sample count
    classes = list(range(43))
    sample_counts = [class_counts.get(cls, 0) for cls in classes]
    
    # Categorize by sample count (as proxy for difficulty)
    low_sample_classes = [cls for cls in classes if class_counts.get(cls, 0) < 100]
    medium_sample_classes = [cls for cls in classes if 100 <= class_counts.get(cls, 0) < 500]
    high_sample_classes = [cls for cls in classes if class_counts.get(cls, 0) >= 500]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Sample count distribution
    ax1.hist(sample_counts, bins=20, alpha=0.7, color='skyblue')
    ax1.set_title('Distribution of Sample Counts per Class')
    ax1.set_xlabel('Number of Samples')
    ax1.set_ylabel('Number of Classes')
    ax1.grid(True, alpha=0.3)
    
    # 2. Classes by sample count category
    categories = ['Low (<100)', 'Medium (100-500)', 'High (>500)']
    category_counts = [len(low_sample_classes), len(medium_sample_classes), len(high_sample_classes)]
    ax2.pie(category_counts, labels=categories, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Classes by Sample Count Category')
    
    # 3. Sample count per class
    colors = ['red' if count < 100 else 'orange' if count < 500 else 'green' 
                for count in sample_counts]
    ax3.bar(classes, sample_counts, color=colors, alpha=0.7)
    ax3.set_title('Sample Count per Class (Red: <100, Orange: 100-500, Green: >500)')
    ax3.set_xlabel('Traffic Sign Class')
    ax3.set_ylabel('Number of Samples')
    
    # 4. Class imbalance visualization
    sorted_counts = sorted(sample_counts)
    ax4.plot(range(len(sorted_counts)), sorted_counts, marker='o', markersize=4)
    ax4.set_title('Class Imbalance Curve')
    ax4.set_xlabel('Class Rank (sorted by sample count)')
    ax4.set_ylabel('Number of Samples')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(Config.CHARTS_FOLDER, 'class_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def plot_feature_importance_analysis(self):
    """Chart 5: Feature extraction analysis"""
    # Analyze feature extraction results
    signs_with_features = self.session.query(TrafficSign).filter(
        TrafficSign.hog_features.isnot(None)
    ).all()
    
    if not signs_with_features:
        return None
    
    hog_sizes = []
    haar_sizes = []
    hue_sizes = []
    
    for sign in signs_with_features[:100]:  # Sample first 100
        try:
            if sign.hog_features:
                hog_feat = deserialize_features(sign.hog_features)
                hog_sizes.append(len(hog_feat))
            
            if sign.haar_features:
                haar_feat = deserialize_features(sign.haar_features)
                haar_sizes.append(len(haar_feat))
            
            if sign.hue_histogram:
                hue_feat = deserialize_features(sign.hue_histogram)
                hue_sizes.append(len(hue_feat))
        except:
            continue
    
    # Create feature analysis plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Feature vector sizes
    feature_types = ['HOG', 'Haar', 'Hue Histogram']
    avg_sizes = [np.mean(hog_sizes) if hog_sizes else 0,
                np.mean(haar_sizes) if haar_sizes else 0,
                np.mean(hue_sizes) if hue_sizes else 0]
    
    ax1.bar(feature_types, avg_sizes, color=['blue', 'green', 'red'])
    ax1.set_title('Average Feature Vector Sizes')
    ax1.set_ylabel('Feature Vector Length')
    
    # 2. Feature extraction coverage
    total_signs = self.session.query(TrafficSign).count()
    signs_with_hog = self.session.query(TrafficSign).filter(TrafficSign.hog_features.isnot(None)).count()
    signs_with_haar = self.session.query(TrafficSign).filter(TrafficSign.haar_features.isnot(None)).count()
    signs_with_hue = self.session.query(TrafficSign).filter(TrafficSign.hue_histogram.isnot(None)).count()
    
    coverage = [signs_with_hog/total_signs*100, signs_with_haar/total_signs*100, 
                signs_with_hue/total_signs*100]
    
    ax2.bar(feature_types, coverage, color=['blue', 'green', 'red'], alpha=0.7)
    ax2.set_title('Feature Extraction Coverage (%)')
    ax2.set_ylabel('Percentage of Images')
    ax2.set_ylim(0, 100)
    
    # 3. Dataset statistics
    train_count = self.session.query(TrafficSign).filter(TrafficSign.is_training == True).count()
    test_count = self.session.query(TrafficSign).filter(TrafficSign.is_training == False).count()
    
    ax3.pie([train_count, test_count], labels=['Training', 'Test'], autopct='%1.1f%%')
    ax3.set_title('Training vs Test Data Split')
    
    # 4. Processing timeline (mock data for illustration)
    ax4.text(0.5, 0.5, f'Dataset Statistics:\n\n'
                        f'Total Images: {total_signs}\n'
                        f'Training Images: {train_count}\n'
                        f'Test Images: {test_count}\n'
                        f'Classes: 43 (0-42)\n'
                        f'Feature Types: 3\n'
                        f'Images with Features: {signs_with_hog}',
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Dataset Summary')
    
    plt.tight_layout()
    plot_path = os.path.join(self.output_dir, 'feature_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def generate_all_charts():
    """Generate all charts"""
    charts = {}
    
    print("Generating charts...")
    
    charts['class_distribution'] = plot_class_distribution()
    charts['model_comparison'] = plot_model_comparison()
    charts['hyperparameter_analysis'] = plot_hyperparameter_analysis()
    charts['class_analysis'] = plot_class_performance_analysis()
    charts['feature_analysis'] = plot_feature_importance_analysis()
    
    print("All charts generated!")
    return charts