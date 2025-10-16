from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
from modules.utils import ExcelExporter
import pandas as pd
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.ml.tuning import CrossValidatorModel, TrainValidationSplitModel

class ModelTrainer:
    def __init__(self, spark_session):
        self.spark = spark_session
        self.model = None
        self.predictions = None
        self.exporter = ExcelExporter()
    
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
        # Xuất predictions
        predictions_pd = self.predictions.toPandas()
        self.exporter.export_step_data(predictions_pd, "04_model_predictions")
        
        # Xuất model metrics
        metrics_data = pd.DataFrame([{
            'metric': 'RMSE',
            'value': rmse,
            'rank': params['rank'],
            'maxIter': params['maxIter'],
            'regParam': params['regParam']
        }])
        self.exporter.export_step_data(metrics_data, "04_model_metrics")
        
        return self.model, rmse

    def tune_model(self, ratings_df, param_grid=None, num_folds=3, use_crossval=True, parallelism=2):
        """Hyperparameter tuning for ALS using CrossValidator or TrainValidationSplit.
        ratings_df: Spark DataFrame or pandas DataFrame (will be converted).
        Returns best_model (ALSModel) and a summary dict.
        """
        print("Starting hyperparameter tuning...")
        # convert pandas -> spark if needed
        if isinstance(ratings_df, pd.DataFrame):
            spark_ratings = self.create_spark_dataframe(ratings_df)
        else:
            spark_ratings = ratings_df

        # base ALS estimator
        from pyspark.ml.recommendation import ALS
        als = ALS(
            userCol="user_id",
            itemCol="product_id",
            ratingCol="rating",
            coldStartStrategy="drop",
            nonnegative=True
        )

        # default grid if none provided
        if param_grid is None:
            param_grid = ParamGridBuilder() \
                .addGrid(als.rank, [8, 10, 12]) \
                .addGrid(als.maxIter, [5, 10]) \
                .addGrid(als.regParam, [0.05, 0.1]) \
                .build()
        else:
            # assume user passed a list of dicts or built ParamGridBuilder().build()
            if not isinstance(param_grid, list):
                param_grid = ParamGridBuilder().build()  # fallback to avoid crash

        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

        if use_crossval:
            cv = CrossValidator(
                estimator=als,
                estimatorParamMaps=param_grid,
                evaluator=evaluator,
                numFolds=num_folds,
                parallelism=parallelism
            )
            cv_model = cv.fit(spark_ratings)
            best_model = cv_model.bestModel
            # try to estimate best RMSE from avgMetrics (lowest)
            avg_metrics = cv_model.avgMetrics if hasattr(cv_model, 'avgMetrics') else []
            best_metric = min(avg_metrics) if avg_metrics else None
            summary = {'method': 'crossval', 'numModels': len(param_grid), 'best_metric_rmse': float(best_metric) if best_metric is not None else None}
        else:
            tvs = TrainValidationSplit(
                estimator=als,
                estimatorParamMaps=param_grid,
                evaluator=evaluator,
                trainRatio=0.8,
                parallelism=parallelism
            )
            tvs_model = tvs.fit(spark_ratings)
            best_model = tvs_model.bestModel
            avg_metrics = tvs_model.validationMetrics if hasattr(tvs_model, 'validationMetrics') else []
            best_metric = min(avg_metrics) if avg_metrics else None
            summary = {'method': 'tvs', 'numModels': len(param_grid), 'best_metric_rmse': float(best_metric) if best_metric is not None else None}

        # store and return
        self.model = best_model
        print("✅ Hyperparameter tuning finished. Best model stored in trainer.model")
        print(f"Summary: {summary}")
        return best_model, summary

    def recommend_for_user(self, user_id, n=10):
        """Return pandas DataFrame of recommendations for a single user using the stored model."""
        if self.model is None:
            raise RuntimeError("No model loaded - train or load a model first.")
        user_df = self.spark.createDataFrame([(int(user_id),)], ["user_id"])
        rec = self.model.recommendForUserSubset(user_df, n)
        rec_pd = rec.toPandas()
        rows = []
        for _, r in rec_pd.iterrows():
            for rec_item in r['recommendations']:
                rows.append({'user_id': r['user_id'], 'product_id': rec_item['product_id'], 'predicted_rating': rec_item['rating']})
        return pd.DataFrame(rows)

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