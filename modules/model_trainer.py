from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

class ModelTrainer:
    def __init__(self, spark_session):
        self.spark = spark_session
        self.model = None
        self.predictions = None
    
    def create_spark_dataframe(self, ratings_df):
        """Convert pandas DataFrame to Spark DataFrame"""
        print("Creating Spark DataFrame...")
        spark_ratings = self.spark.createDataFrame(
            ratings_df[['user_id', 'product_id', 'rating']]
        )
        spark_ratings.cache()
        print(f"Spark DataFrame count: {spark_ratings.count()}")
        return spark_ratings
    
    def train_model(self, ratings_df, params):
        """Train ALS model with given parameters"""
        print(f"Training ALS model with params: {params}")
        
        # Split data
        (training, test) = ratings_df.randomSplit([0.8, 0.2], seed=42)
        
        # Build and train model
        als = ALS(
            maxIter=params['maxIter'],
            rank=params['rank'],
            regParam=params['regParam'],
            userCol="user_id",
            itemCol="product_id",
            ratingCol="rating",
            coldStartStrategy="drop",
            nonnegative=params.get('nonnegative', True)
        )
        
        self.model = als.fit(training)
        
        # Evaluate model
        evaluator = RegressionEvaluator(
            metricName="rmse", 
            labelCol="rating", 
            predictionCol="prediction"
        )
        
        predictions = self.model.transform(test)
        self.predictions = predictions.filter(col("prediction").isNotNull())
        rmse = evaluator.evaluate(self.predictions)
        
        print(f"Root Mean Square Error (RMSE) = {rmse:.4f}")
        return self.model, rmse
    
    def save_model(self, model_path):
        """Save trained model"""
        if self.model:
            self.model.write().overwrite().save(model_path)
            print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load trained model"""
        from pyspark.ml.recommendation import ALSModel
        self.model = ALSModel.load(model_path)
        print(f"Model loaded from {model_path}")
        return self.model