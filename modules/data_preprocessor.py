import pandas as pd
import numpy as np
from modules.utils import ExcelExporter

class DataPreprocessor:
    def __init__(self):
        self.df = None
        self.ratings_df = None
        self.exporter = ExcelExporter()
    
    def preprocess_data(self, df):
        """Clean and preprocess the raw data"""
        print("Preprocessing data...")
        
        # Data cleaning
        self.df = df.dropna(subset=['reviews', 'rating', 'brand'])
        self.df = self.df[self.df['reviews'] > 0]
        
        print(f"After preprocessing: {self.df.shape}")
        self.exporter.export_step_data(self.df, "02_cleaned_data")
        return self.df
    
    def create_user_simulation(self, df, n_users=500):
        """Simulate user interactions with more randomness"""
        print("Creating user simulation with random interactions...")
        
        np.random.seed(42)
        user_ids = list(range(1, n_users + 1))
        user_product_data = []
        
        # Tạo distribution ngẫu nhiên cho số lượng interactions per product
        for idx, product in df.iterrows():
            # Random hơn: base trên reviews nhưng thêm randomness
            base_interactions = max(1, int(product['reviews'] / 10))
            # Thêm variation: -30% đến +50% số interactions
            variation = np.random.uniform(0.7, 1.5)
            n_interactions = max(1, int(base_interactions * variation))
            
            # Đảm bảo không vượt quá số users
            n_interactions = min(n_interactions, len(user_ids))
            
            # Chọn users ngẫu nhiên
            product_users = np.random.choice(
                user_ids, 
                size=n_interactions, 
                replace=False
            )
            
            for user_id in product_users:
                base_rating = product['rating']
                
                # Thêm nhiều randomness vào rating
                if np.random.random() < 0.8:  # 80% rating gần với base rating
                    noise = np.random.normal(0, 0.5)
                    rating = max(1, min(5, base_rating + noise))
                else:  # 20% rating hoàn toàn ngẫu nhiên
                    rating = np.random.uniform(1, 5)
                
                # Làm tròn rating
                rating = round(rating, 1)
                
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
        
        # THÊM: Tạo thêm một số random cross-interactions
        # Mỗi user có thêm một số interactions ngẫu nhiên với products khác
        n_cross_interactions = int(len(user_product_data) * 0.2)  # 20% thêm
        
        for _ in range(n_cross_interactions):
            user_id = np.random.choice(user_ids)
            product_idx = np.random.randint(0, len(df))
            product = df.iloc[product_idx]
            
            # Rating hoàn toàn ngẫu nhiên cho cross-interactions
            rating = np.random.uniform(1, 5)
            rating = round(rating, 1)
            
            user_product_data.append({
                'user_id': user_id,
                'product_id': product_idx + 1,
                'product_name': product['product_name'],
                'brand': product['brand'],
                'rating': rating,
                'reviews': product['reviews'],
                'sales_volume': product['sales_volume'],
                'price': product['price']
            })
        
        self.ratings_df = pd.DataFrame(user_product_data)
        
        # Thống kê
        avg_ratings = self.ratings_df['rating'].mean()
        rating_std = self.ratings_df['rating'].std()
        
        print(f"Generated {len(self.ratings_df)} interactions")
        print(f"Unique users: {self.ratings_df['user_id'].nunique()}")
        print(f"Unique products: {self.ratings_df['product_id'].nunique()}")
        print(f"Average rating: {avg_ratings:.2f} ± {rating_std:.2f}")
        
        # Xuất thống kê distribution
        user_stats = self.ratings_df.groupby('user_id').agg({
            'product_id': 'count',
            'rating': 'mean'
        }).rename(columns={'product_id': 'n_interactions', 'rating': 'avg_rating'})
        
        product_stats = self.ratings_df.groupby('product_id').agg({
            'user_id': 'count', 
            'rating': 'mean'
        }).rename(columns={'user_id': 'n_ratings', 'rating': 'avg_rating'})
        
        print(f"User interactions stats: min={user_stats['n_interactions'].min()}, "
            f"max={user_stats['n_interactions'].max()}, avg={user_stats['n_interactions'].mean():.1f}")
        print(f"Product ratings stats: min={product_stats['n_ratings'].min()}, "
            f"max={product_stats['n_ratings'].max()}, avg={product_stats['n_ratings'].mean():.1f}")

        self.exporter.export_step_data(self.ratings_df, "03_user_product_matrix")
        
        return self.ratings_df