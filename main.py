from pyspark.sql import SparkSession
from config import Config
from modules.data_loader import DataLoader
from modules.data_preprocessor import DataPreprocessor
from modules.model_trainer import ModelTrainer
from modules.recommender import Recommender
from modules.evaluator import Evaluator
from modules.visualizer import Visualizer
from modules.utils import Utils
import pandas as pd
import os

def setup_spark():
    """Initialize Spark session"""
    return SparkSession.builder \
        .appName(Config.SPARK_CONFIG["appName"]) \
        .config("spark.sql.adaptive.enabled", Config.SPARK_CONFIG["spark.sql.adaptive.enabled"]) \
        .config("spark.sql.adaptive.coalescePartitions.enabled", Config.SPARK_CONFIG["spark.sql.adaptive.coalescePartitions.enabled"]) \
        .config("spark.local.dir", Config.SPARK_CONFIG["spark.local.dir"]) \
        .getOrCreate()

def run_data_preprocessing():
    """Module 1: Data preprocessing"""
    print("=== MODULE 1: DATA PREPROCESSING ===")
    spark = setup_spark()
    
    # Load data
    loader = DataLoader(spark)
    raw_data = loader.load_raw_data(Config.DATA_PATHS['raw_data'])
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_data(raw_data)
    ratings_data = preprocessor.create_user_simulation(processed_data)
    
    # Save processed data
    loader.df = processed_data
    loader.ratings_df = ratings_data
    loader.save_processed_data(Config.DATA_PATHS['processed_data'])
    loader.save_ratings_data(Config.DATA_PATHS['ratings_data'])
    
    spark.stop()
    return processed_data, ratings_data

def run_model_training():
    """Module 2: Model training"""
    print("\n=== MODULE 2: MODEL TRAINING ===")
    spark = setup_spark()
    
    # Load preprocessed data
    loader = DataLoader(spark)
    ratings_data = loader.load_ratings_data(Config.DATA_PATHS['ratings_data'])
    
    # Train model
    trainer = ModelTrainer(spark)
    spark_ratings = trainer.create_spark_dataframe(ratings_data)
    model, rmse = trainer.train_model(spark_ratings, Config.MODEL_PARAMS)
    
    # Save model
    trainer.save_model(Config.DATA_PATHS['model_path'])
    
    spark.stop()
    return model, rmse, ratings_data

def run_recommendation_generation():
    """Module 3: Recommendation generation"""
    print("\n=== MODULE 3: RECOMMENDATION GENERATION ===")
    spark = setup_spark()
    
    # Load data and model
    loader = DataLoader(spark)
    processed_data = loader.load_processed_data(Config.DATA_PATHS['processed_data'])
    ratings_data = loader.load_ratings_data(Config.DATA_PATHS['ratings_data'])
    
    trainer = ModelTrainer(spark)
    model = trainer.load_model(Config.DATA_PATHS['model_path'])
    
    # Generate recommendations
    recommender = Recommender(model, spark)
    recommendations = recommender.generate_recommendations(n_recommendations=10)
    final_recommendations = recommender.add_product_info(recommendations, processed_data)
    
    # Save recommendations
    recommender.save_recommendations(final_recommendations, Config.DATA_PATHS['recommendations_path'])
    
    spark.stop()
    return final_recommendations, processed_data, ratings_data, model

def run_evaluation_and_visualization():
    """Module 4: Evaluation and visualization"""
    print("\n=== MODULE 4: EVALUATION AND VISUALIZATION ===")
    spark = setup_spark()
    
    # Load data
    loader = DataLoader(spark)
    recommendations = pd.read_csv(Config.DATA_PATHS['recommendations_path'])
    processed_data = loader.load_processed_data(Config.DATA_PATHS['processed_data'])
    ratings_data = loader.load_ratings_data(Config.DATA_PATHS['ratings_data'])
    
    trainer = ModelTrainer(spark)
    model = trainer.load_model(Config.DATA_PATHS['model_path'])
    
    # Evaluation
    evaluator = Evaluator(model, spark)
    precision_at_10 = evaluator.calculate_precision_at_k(
        ratings_data, 
        k=Config.EVALUATION['k'],
        min_rating=Config.EVALUATION['min_rating']
    )
    
    business_analysis = evaluator.business_value_analysis(
        recommendations, ratings_data, processed_data
    )
    
    # Visualization
    visualizer = Visualizer()
    visualizer.create_recommendation_plots(recommendations)
    network = visualizer.create_recommendation_network(recommendations)
    visualizer.export_network_gephi(network, "results/recommendation_network.gexf")
    
    # Generate report
    metrics = {'rmse': 0.0, 'precision_at_10': precision_at_10}  # RMSE would come from training
    report_data = Utils.generate_report(metrics, business_analysis, ratings_data, None)
    Utils.save_report(report_data, "results/evaluation_report.json")
    
    spark.stop()

def run_complete_pipeline():
    """Run complete pipeline"""
    # Module 1: Data preprocessing
    processed_data, ratings_data = run_data_preprocessing()
    
    # Module 2: Model training
    model, rmse, ratings_data = run_model_training()
    
    # Module 3: Recommendation generation
    recommendations, processed_data, ratings_data, model = run_recommendation_generation()
    
    # Module 4: Evaluation and visualization
    run_evaluation_and_visualization()
    
    print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Run specific module or complete pipeline
    run_complete_pipeline()  # Or call individual modules