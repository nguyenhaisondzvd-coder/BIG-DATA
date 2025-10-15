# Configuration settings
import os

class Config:
    # Spark configuration
    SPARK_CONFIG = {
        "appName": "ProductRecommendationALS",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.local.dir": "temp"
    }
    
    # Model parameters
    MODEL_PARAMS = {
        'rank': 10,
        'maxIter': 10,
        'regParam': 0.1,
        'nonnegative': True
    }
    
    # Data paths
    DATA_PATHS = {
        'raw_data': 'data/1.shope_lazada.xlsx',
        'processed_data': 'results/processed_data.csv',
        'ratings_data': 'results/ratings_data.csv',
        'model_path': 'models/als_model',
        'recommendations_path': 'results/recommendations.csv'
    }
    
    # Evaluation parameters
    EVALUATION = {
        'test_size': 0.2,
        'k': 10,
        'min_rating': 4.0
    }

    # HDFS Configuration
    HDFS_CONFIG = {
        'base_url': 'hdfs://namenode:9000',
        'data_path': '/user/recommendation/data',
        'model_path': '/user/recommendation/models',
        'results_path': '/user/recommendation/results'
    }