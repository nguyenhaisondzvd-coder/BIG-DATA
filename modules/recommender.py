import pandas as pd
from pyspark.sql.functions import col, explode

class Recommender:
    def __init__(self, model, spark_session):
        self.model = model
        self.spark = spark_session
        self.user_recs = None
    
    def generate_recommendations(self, n_recommendations=10):
        """Generate recommendations for all users"""
        print(f"Generating {n_recommendations} recommendations per user...")
        
        self.user_recs = self.model.recommendForAllUsers(n_recommendations)
        
        # Convert to pandas
        user_recs_pd = self.user_recs.toPandas()
        
        recommendations = []
        for _, row in user_recs_pd.iterrows():
            user_id = row['user_id']
            for rec in row['recommendations']:
                recommendations.append({
                    'user_id': user_id,
                    'product_id': rec['product_id'],
                    'predicted_rating': rec['rating']
                })
        
        return pd.DataFrame(recommendations)
    
    def get_recommendations_for_user(self, user_id, n_recommendations=10):
        """Get recommendations for specific user"""
        user_df = self.spark.createDataFrame([(user_id,)], ["user_id"])
        user_recs = self.model.recommendForUserSubset(user_df, n_recommendations)
        return user_recs
    
    def add_product_info(self, recommendations, product_df):
        """Add product information to recommendations"""
        product_info = product_df.reset_index()[['index', 'product_name', 'brand', 'product_type', 'price']]
        product_info = product_info.rename(columns={'index': 'product_id'})
        product_info['product_id'] = product_info['product_id'] + 1
        
        final_recommendations = recommendations.merge(
            product_info, 
            on='product_id', 
            how='left'
        )
        
        print(f"Final recommendations with product info: {len(final_recommendations)}")
        return final_recommendations
    
    def filter_by_brand(self, recommendations, preferred_brands):
        """Filter recommendations by brand"""
        filtered_recs = recommendations[
            recommendations['brand'].isin(preferred_brands)
        ]
        print(f"Filtered from {len(recommendations)} to {len(filtered_recs)} recommendations")
        return filtered_recs
    
    def save_recommendations(self, recommendations, file_path):
        """Save recommendations to CSV"""
        recommendations.to_csv(file_path, index=False)
        print(f"Recommendations saved to {file_path}")