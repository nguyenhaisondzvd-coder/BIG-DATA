import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self):
        self.df = None
        self.ratings_df = None
    
    def preprocess_data(self, df):
        """Clean and preprocess the raw data"""
        print("Preprocessing data...")
        
        # Data cleaning
        self.df = df.dropna(subset=['reviews', 'rating', 'brand'])
        self.df = self.df[self.df['reviews'] > 0]
        
        print(f"After preprocessing: {self.df.shape}")
        return self.df
    
    def create_user_simulation(self, df, n_users=500):
        """Simulate user interactions"""
        print("Creating user simulation...")
        
        np.random.seed(42)
        user_ids = list(range(1, n_users + 1))
        user_product_data = []
        
        for idx, product in df.iterrows():
            n_interactions = max(1, int(product['reviews'] / 10))
            product_users = np.random.choice(
                user_ids, 
                size=min(n_interactions, len(user_ids)), 
                replace=False
            )
            
            for user_id in product_users:
                base_rating = product['rating']
                noise = np.random.normal(0, 0.5)
                rating = max(1, min(5, base_rating + noise))
                
                user_product_data.append({
                    'user_id': user_id,
                    'product_id': idx + 1,
                    'product_name': product['product_name'],
                    'brand': product['brand'],
                    'rating': rating,
                    'reviews': product['reviews'],
                    'sales_volume': product['sales_volume'],
                    'price': product['price']
                })
        
        self.ratings_df = pd.DataFrame(user_product_data)
        print(f"Generated {len(self.ratings_df)} interactions")
        print(f"Unique users: {self.ratings_df['user_id'].nunique()}")
        print(f"Unique products: {self.ratings_df['product_id'].nunique()}")
        
        return self.ratings_df