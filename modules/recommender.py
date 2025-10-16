import pandas as pd
import numpy as np
from pyspark.sql.functions import col, explode
from modules.utils import ExcelExporter

class Recommender:
    def __init__(self, model, spark_session):
        self.model = model
        self.spark = spark_session
        self.user_recs = None
        self.exporter = ExcelExporter()
    
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
        
        rec_df = pd.DataFrame(recommendations)
        self.exporter.export_step_data(rec_df, "05_raw_recommendations")
        return rec_df
    
    def add_product_info(self, recommendations, product_df):
        """Add product information to recommendations"""
        product_info = product_df.reset_index()[['index', 'product_name', 'brand', 'product_type', 'price', 'rating', 'reviews']]
        product_info = product_info.rename(columns={'index': 'product_id'})
        product_info['product_id'] = product_info['product_id'] + 1
        
        final_recommendations = recommendations.merge(
            product_info, 
            on='product_id', 
            how='left'
        )
        
        print(f"Final recommendations with product info: {len(final_recommendations)}")
        self.exporter.export_step_data(final_recommendations, "06_final_recommendations")
        return final_recommendations
    
    def enhance_recommendations_with_similar_products(self, recommendations, product_df, ratings_df, top_k_similar=5):
        """Enhance recommendations by adding similar products based on product_type and rating/reviews"""
        print("Enhancing recommendations with similar products...")
        
        enhanced_recs = []
        
        for _, rec in recommendations.iterrows():
            # Add original recommendation
            enhanced_recs.append(rec.to_dict())
            
            # Find similar products
            original_product = product_df[product_df.index == rec['product_id'] - 1]
            if len(original_product) > 0:
                orig_prod = original_product.iloc[0]
                similar_products = self._find_similar_products(
                    orig_prod, product_df, ratings_df, top_k_similar
                )
                
                # Add similar products with slightly lower predicted rating
                for i, similar_prod in similar_products.iterrows():
                    similar_rec = rec.copy()
                    similar_rec['product_id'] = similar_prod.name + 1
                    similar_rec['product_name'] = similar_prod['product_name']
                    similar_rec['brand'] = similar_prod['brand']
                    similar_rec['product_type'] = similar_prod['product_type']
                    similar_rec['price'] = similar_prod['price']
                    similar_rec['rating'] = similar_prod['rating']
                    similar_rec['reviews'] = similar_prod['reviews']
                    # Slightly lower predicted rating for similar products
                    similar_rec['predicted_rating'] = rec['predicted_rating'] * np.random.uniform(0.85, 0.95)
                    similar_rec['similarity_source'] = 'similar_product'
                    
                    enhanced_recs.append(similar_rec)
        
        enhanced_df = pd.DataFrame(enhanced_recs)
        # Remove duplicates and sort by user_id and predicted_rating
        enhanced_df = enhanced_df.drop_duplicates(subset=['user_id', 'product_id'])
        enhanced_df = enhanced_df.sort_values(['user_id', 'predicted_rating'], ascending=[True, False])
        
        print(f"Enhanced recommendations: {len(enhanced_df)} (from {len(recommendations)})")
        self.exporter.export_step_data(enhanced_df, "06_enhanced_recommendations")
        return enhanced_df
    
    def _find_similar_products(self, target_product, product_df, ratings_df, top_k=5):
        """Find similar products based on product_type, rating, and reviews"""
        same_type_products = product_df[
            (product_df['product_type'] == target_product['product_type']) &
            (product_df.index != target_product.name)  # exclude original product
        ].copy()
        
        if len(same_type_products) == 0:
            return pd.DataFrame()
        
        # Calculate similarity score
        target_rating = target_product['rating']
        target_reviews = target_product['reviews']
        target_price = target_product['price']
        
        same_type_products['rating_similarity'] = 1 - abs(same_type_products['rating'] - target_rating) / 4.0
        same_type_products['reviews_similarity'] = 1 - abs(same_type_products['reviews'] - target_reviews) / max(target_reviews, same_type_products['reviews'].max())
        same_type_products['price_similarity'] = 1 - abs(same_type_products['price'] - target_price) / max(target_price, same_type_products['price'].max())
        
        # Combined similarity score
        same_type_products['similarity_score'] = (
            same_type_products['rating_similarity'] * 0.4 +
            same_type_products['reviews_similarity'] * 0.3 +
            same_type_products['price_similarity'] * 0.3
        )
        
        # Return top similar products
        return same_type_products.nlargest(top_k, 'similarity_score')
    
    def filter_by_brand(self, recommendations, preferred_brands):
        """Filter recommendations by brand"""
        filtered_recs = recommendations[
            recommendations['brand'].isin(preferred_brands)
        ]
        print(f"Filtered from {len(recommendations)} to {len(filtered_recs)} recommendations")
        self.exporter.export_step_data(filtered_recs, "07_brand_filtered_recommendations")
        return filtered_recs
    
    def get_recommendations_for_user(self, user_id, n_recommendations=10):
        """Get recommendations for specific user"""
        user_df = self.spark.createDataFrame([(user_id,)], ["user_id"])
        user_recs = self.model.recommendForUserSubset(user_df, n_recommendations)
        return user_recs
    
    def save_recommendations(self, recommendations, file_path):
        """Save recommendations to CSV"""
        recommendations.to_csv(file_path, index=False)
        print(f"Recommendations saved to {file_path}")