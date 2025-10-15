from pyspark.sql.functions import col, explode
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
from modules.utils import ExcelExporter

class Evaluator:
    def __init__(self, model, spark_session):
        self.model = model
        self.spark = spark_session
        self.exporter = ExcelExporter()
    
    def calculate_precision_at_k(self, ratings_df, k=10, min_rating=4.0):
        """Calculate Precision@K metric"""
        print(f"Calculating Precision@{k}...")
        
        # Get top K recommendations
        user_recs = self.model.recommendForAllUsers(k)
        
        # Explode recommendations
        exploded_recs = user_recs.select(
            "user_id", 
            explode("recommendations").alias("rec")
        ).select(
            "user_id", 
            col("rec.product_id").alias("product_id"),
            col("rec.rating").alias("predicted_rating")
        )
        
        # Find actual high ratings
        actual_high_ratings = ratings_df[
            ratings_df['rating'] >= min_rating
        ][['user_id', 'product_id']]
        
        actual_spark = self.spark.createDataFrame(actual_high_ratings)
        
        # Calculate hits
        hits = exploded_recs.join(
            actual_spark, 
            ["user_id", "product_id"], 
            "inner"
        )
        
        total_recommendations = exploded_recs.count()
        total_hits = hits.count()
        
        precision_at_k = total_hits / total_recommendations if total_recommendations > 0 else 0
        print(f"Precision@{k}: {precision_at_k:.4f}")
         # Xuất evaluation results
        eval_data = pd.DataFrame([{
            'metric': f'Precision@{k}',
            'value': precision_at_k,
            'total_recommendations': total_recommendations,
            'total_hits': total_hits,
            'min_rating_threshold': min_rating
        }])
        self.exporter.export_step_data(eval_data, "08_evaluation_metrics")
        return precision_at_k
    
    def business_value_analysis(self, recommendations, ratings_df, product_df):
        """Analyze business value"""
        print("Analyzing business value...")
        
        avg_price = product_df['price'].mean()
        conversion_rate = 0.05
        
        potential_revenue = len(recommendations) * avg_price * conversion_rate
        brand_diversity = recommendations['brand'].nunique()
        product_diversity = recommendations['product_name'].nunique()
        
        total_users = ratings_df['user_id'].nunique()
        users_with_recs = recommendations['user_id'].nunique()
        user_coverage = users_with_recs / total_users * 100
        
        print(f"Potential revenue: ${potential_revenue:,.2f}")
        print(f"Brand diversity: {brand_diversity}")
        print(f"Product diversity: {product_diversity}")
        print(f"User coverage: {user_coverage:.1f}%")
        
        business_data = {
            'potential_revenue': potential_revenue,
            'brand_diversity': brand_diversity,
            'product_diversity': product_diversity,
            'user_coverage': user_coverage
        }
        
        # Xuất business analysis
        business_df = pd.DataFrame([business_data])
        self.exporter.export_step_data(business_df, "09_business_analysis")
        
        return business_data