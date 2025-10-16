from pyspark.sql import SparkSession
from config import Config
from modules.data_loader import DataLoader
from modules.data_preprocessor import DataPreprocessor
from modules.model_trainer import ModelTrainer
from modules.recommender import Recommender
from modules.evaluator import Evaluator
from modules.visualizer import Visualizer
from modules.utils import Utils, HDFSManager, ExcelExporter
import pandas as pd
import os
import time
import subprocess

def setup_spark_cluster():
    """Initialize Spark session for cluster mode"""
    print("Initializing Spark session for cluster...")
    
    return SparkSession.builder \
        .appName("ProductRecommendationCluster") \
        .master("spark://spark-master:7077") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.executor.memory", "1g") \
        .config("spark.driver.memory", "1g") \
        .config("spark.executor.cores", "2") \
        .config("spark.network.timeout", "300s") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
        .config("spark.sql.warehouse.dir", "hdfs://namenode:9000/user/spark/warehouse") \
        .getOrCreate()

def setup_spark_standalone():
    """Initialize Spark session for standalone mode"""
    print("Initializing Spark session for standalone...")
    
    return SparkSession.builder \
        .appName("ProductRecommendation") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.executor.memory", "1g") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()

def setup_hdfs_environment():
    """Setup HDFS environment before running pipeline"""
    print("Setting up HDFS environment...")
    
    hdfs_manager = HDFSManager()
    
    try:
        # Create necessary directories in HDFS
        hdfs_manager.setup_hdfs_directories()
        
        # Upload data to HDFS if exists
        if os.path.exists(Config.DATA_PATHS['raw_data']):
            hdfs_manager.upload_to_hdfs(
                Config.DATA_PATHS['raw_data'],
                '/user/recommendation/data/raw_data.xlsx'
            )
            print("✅ Raw data uploaded to HDFS")
        else:
            print("⚠️ Raw data file not found, skipping HDFS upload")
            
    except Exception as e:
        print(f"⚠️ HDFS setup warning: {e}")
        print("Continuing with local file system...")

def wait_for_hdfs():
    """Wait for HDFS to be ready"""
    print("Waiting for HDFS to be ready...")
    hdfs_manager = HDFSManager()
    
    max_retries = 30
    for i in range(max_retries):
        try:
            result = subprocess.run(
                ["hdfs", "dfs", "-ls", "/"],
                capture_output=True,
                text=True,
                check=True
            )
            print("✅ HDFS is ready!")
            return True
        except subprocess.CalledProcessError:
            print(f"Attempt {i+1}/{max_retries}: HDFS not ready yet...")
            time.sleep(5)
    
    print("❌ HDFS not available, continuing with local storage...")
    return False

def wait_for_spark():
    """Wait for Spark Master to be ready"""
    print("Waiting for Spark Master to be ready...")
    
    max_retries = 30
    for i in range(max_retries):
        try:
            # Try to create a Spark session
            spark = SparkSession.builder \
                .appName("HealthCheck") \
                .master("spark://spark-master:7077") \
                .config("spark.network.timeout", "300s") \
                .config("spark.executor.heartbeatInterval", "60s") \
                .getOrCreate()
            spark.stop()
            print("✅ Spark Master is ready!")
            return True
        except Exception as e:
            print(f"Attempt {i+1}/{max_retries}: Spark Master not ready yet...")
            time.sleep(10)
    
    print("❌ Spark Master not available, using standalone mode...")
    return False

def run_data_preprocessing():
    """Module 1: Data preprocessing"""
    print("\n" + "="*50)
    print("MODULE 1: DATA PREPROCESSING")
    print("="*50)
    
    # Try cluster mode first, fallback to standalone
    try:
        spark = setup_spark_cluster()
        print("✅ Running in Spark Cluster mode")
    except Exception as e:
        print(f"⚠️ Cluster mode failed: {e}")
        spark = setup_spark_standalone()
        print("✅ Running in Spark Standalone mode")
    
    try:
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
        
        # Upload to HDFS if available
        try:
            hdfs_manager = HDFSManager()
            hdfs_manager.upload_to_hdfs(
                Config.DATA_PATHS['processed_data'],
                '/user/recommendation/data/processed_data.csv'
            )
            hdfs_manager.upload_to_hdfs(
                Config.DATA_PATHS['ratings_data'],
                '/user/recommendation/data/ratings_data.csv'
            )
        except Exception as e:
            print(f"⚠️ HDFS upload skipped: {e}")
        
        return processed_data, ratings_data
        
    finally:
        spark.stop()

def run_model_training():
    """Module 2: Model training"""
    print("\n" + "="*50)
    print("MODULE 2: MODEL TRAINING")
    print("="*50)
    
    # Try cluster mode first, fallback to standalone
    try:
        spark = setup_spark_cluster()
        print("✅ Running in Spark Cluster mode")
    except Exception as e:
        print(f"⚠️ Cluster mode failed: {e}")
        spark = setup_spark_standalone()
        print("✅ Running in Spark Standalone mode")
    
    try:
        # Load preprocessed data
        loader = DataLoader(spark)
        ratings_data = loader.load_ratings_data(Config.DATA_PATHS['ratings_data'])
        
        # Train model
        trainer = ModelTrainer(spark)
        spark_ratings = trainer.create_spark_dataframe(ratings_data)
        model, rmse = trainer.train_model(spark_ratings, Config.MODEL_PARAMS)
        
        # Save model
        trainer.save_model(Config.DATA_PATHS['model_path'])
        
        print(f"✅ Model training completed with RMSE: {rmse:.4f}")
        return model, rmse, ratings_data
        
    finally:
        spark.stop()

def run_recommendation_generation():
    """Module 3: Recommendation generation"""
    print("\n" + "="*50)
    print("MODULE 3: RECOMMENDATION GENERATION")
    print("="*50)
    
    # Try cluster mode first, fallback to standalone
    try:
        spark = setup_spark_cluster()
        print("✅ Running in Spark Cluster mode")
    except Exception as e:
        print(f"⚠️ Cluster mode failed: {e}")
        spark = setup_spark_standalone()
        print("✅ Running in Spark Standalone mode")
    
    try:
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
        
        # Filter by top brands
        top_brands = processed_data['brand'].value_counts().head(5).index.tolist()
        filtered_recommendations = recommender.filter_by_brand(final_recommendations, top_brands)
        
        # Save recommendations
        recommender.save_recommendations(final_recommendations, Config.DATA_PATHS['recommendations_path'])
        
        print(f"✅ Generated {len(final_recommendations)} recommendations")
        print(f"✅ Filtered to {len(filtered_recommendations)} recommendations by top brands")
        
        return final_recommendations, processed_data, ratings_data, model
        
    finally:
        spark.stop()

def run_evaluation_and_visualization():
    """Module 4: Evaluation and visualization"""
    print("\n" + "="*50)
    print("MODULE 4: EVALUATION AND VISUALIZATION")
    print("="*50)
    
    spark = setup_spark_standalone()  # Use standalone for evaluation (lighter)
    
    try:
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
        network = visualizer.create_recommendation_network(recommendations, sample_users=30)
        visualizer.export_network_gephi(network, "results/recommendation_network.gexf")
        
        # Generate report
        metrics = {'rmse': 0.0, 'precision_at_10': precision_at_10}  # RMSE would come from training
        report_data = Utils.generate_report(metrics, business_analysis, ratings_data, None)
        Utils.save_report(report_data, "results/evaluation_report.json")
        
        print("✅ Evaluation and visualization completed")
        
    finally:
        spark.stop()

def run_real_time_simulation():
    """Module 5: Real-time recommendation simulation"""
    print("\n" + "="*50)
    print("MODULE 5: REAL-TIME SIMULATION")
    print("="*50)
    
    spark = setup_spark_standalone()
    
    try:
        # Load model
        trainer = ModelTrainer(spark)
        model = trainer.load_model(Config.DATA_PATHS['model_path'])
        
        # Real-time simulation
        recommender = Recommender(model, spark)
        
        # Simulate new user with some ratings
        new_user_ratings = [
            (1, 1, 4.5),  # (user_id, product_id, rating)
            (3, 5, 3.8),
            (4, 10, 4.2)
        ]
        
        new_user_df = spark.createDataFrame(new_user_ratings, ["user_id", "product_id", "rating"])
        new_user_recs = model.recommendForUserSubset(new_user_df, 5)
        
        print("Real-time recommendations for new user:")
        new_user_recs.show()
        
        # Export real-time results
        exporter = ExcelExporter()
        real_time_recs = new_user_recs.toPandas()
        exporter.export_step_data(real_time_recs, "10_real_time_recommendations")
        
        print("✅ Real-time simulation completed")
        
    finally:
        spark.stop()

def run_model_tuning():
    """Module: Hyperparameter tuning (Grid Search)"""
    print("\n" + "="*50)
    print("MODULE: HYPERPARAMETER TUNING")
    print("="*50)
    spark = setup_spark_standalone()
    try:
        loader = DataLoader(spark)
        ratings_data = loader.load_ratings_data(Config.DATA_PATHS['ratings_data'])
        trainer = ModelTrainer(spark)
        spark_ratings = trainer.create_spark_dataframe(ratings_data)
        best_model, summary = trainer.tune_model(spark_ratings, num_folds=3, use_crossval=True)
        # Save tuned model
        trainer.model = best_model
        trainer.save_model(Config.DATA_PATHS['model_path'])
        print("✅ Tuning complete and best model saved.")
        return summary
    finally:
        spark.stop()

def run_prediction_cli(user_id, top_k=10):
    """Load model and print/save recommendations for a user (CLI friendly)"""
    print(f"Running prediction for user_id={user_id}, top_k={top_k}")
    spark = setup_spark_standalone()
    try:
        trainer = ModelTrainer(spark)
        trainer.load_model(Config.DATA_PATHS['model_path'])
        recs_df = trainer.recommend_for_user(user_id, n=top_k)
        out_path = f"results/predictions_user_{user_id}.csv"
        recs_df.to_csv(out_path, index=False)
        print(f"✅ Recommendations for user {user_id} saved to {out_path}")
        print(recs_df.head(20).to_string(index=False))
        return recs_df
    finally:
        spark.stop()

def run_complete_pipeline():
    """Run complete Big Data pipeline"""
    print("🚀 STARTING BIG DATA RECOMMENDATION SYSTEM PIPELINE")
    print("📊 This pipeline demonstrates Big Data processing with Spark Cluster + HDFS")
    print("="*70)
    
    # Setup environment
    start_time = time.time()
    
    # Wait for services if in cluster mode
    is_cluster = wait_for_spark()
    is_hdfs = wait_for_hdfs()
    
    if is_cluster and is_hdfs:
        setup_hdfs_environment()
        print("🎯 Running in FULL CLUSTER MODE: Spark Cluster + HDFS")
    else:
        print("🎯 Running in HYBRID MODE: Spark Standalone + Local Storage")
    
    try:
        # Module 1: Data preprocessing
        processed_data, ratings_data = run_data_preprocessing()
        
        # Module 2: Model training
        model, rmse, ratings_data = run_model_training()
        
        # Module 3: Recommendation generation
        recommendations, processed_data, ratings_data, model = run_recommendation_generation()
        
        # Module 4: Evaluation and visualization
        run_evaluation_and_visualization()
        
        # Module 5: Real-time simulation
        run_real_time_simulation()
        
        execution_time = time.time() - start_time
        print("\n" + "="*70)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
        print("="*70)
        
        # Final summary
        print("\n📋 RESULTS SUMMARY:")
        print(f"   • Processed {len(processed_data)} products")
        print(f"   • Generated {len(ratings_data)} user interactions")
        print(f"   • Created recommendation matrix")
        print(f"   • Exported results to HDFS and local storage")
        print(f"   • Generated visualizations and network analysis")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise

def run_individual_module(module_name):
    """Run individual module"""
    import sys
    if module_name == "preprocessing":
        run_data_preprocessing()
    elif module_name == "training":
        run_model_training()
    elif module_name == "recommendation":
        run_recommendation_generation()
    elif module_name == "evaluation":
        run_evaluation_and_visualization()
    elif module_name == "realtime":
        run_real_time_simulation()
    elif module_name == "tune":
        run_model_tuning()
    elif module_name == "predict":
        # CLI usage: python main.py predict <user_id> [top_k]
        if len(sys.argv) < 3:
            print("Usage: python main.py predict <user_id> [top_k]")
            return
        try:
            uid = int(sys.argv[2])
            top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        except ValueError:
            print("user_id and top_k must be integers")
            return
        run_prediction_cli(uid, top_k)
    else:
        print(f"Unknown module: {module_name}")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    
    print("🔧 Big Data Product Recommendation System")
    print("    Built with Spark MLlib ALS + HDFS + Docker Cluster")
    
    # Run complete pipeline or individual module
    import sys
    if len(sys.argv) > 1:
        module_name = sys.argv[1]
        run_individual_module(module_name)
    else:
        run_complete_pipeline()