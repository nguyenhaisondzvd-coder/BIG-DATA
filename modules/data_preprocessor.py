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
        
        # Data cleaning - xử lý price có dấu phẩy
        df_clean = df.copy()
        
        # Chuyển đổi price từ string sang float (xử lý dấu phẩy)
        if 'price' in df_clean.columns:
            df_clean['price'] = df_clean['price'].astype(str).str.replace(',', '.').astype(float)
        
        # Làm sạch dữ liệu
        self.df = df_clean.dropna(subset=['reviews', 'rating', 'brand', 'product_type'])
        self.df = self.df[self.df['reviews'] > 0]
        self.df = self.df[self.df['rating'] > 0]
        
        # Reset index
        self.df = self.df.reset_index(drop=True)
        
        print(f"After preprocessing: {self.df.shape}")
        self.exporter.export_step_data(self.df, "02_cleaned_data")
        return self.df
    
    def create_user_simulation(self, df, n_users=500):
        """Simulate realistic user interactions based on product characteristics"""
        print("Creating realistic user simulation...")
        
        np.random.seed(42)
        user_ids = list(range(1, n_users + 1))
        user_product_data = []
        
        # Tạo user preferences theo product_type
        product_types = df['product_type'].unique()
        user_preferences = {}
        
        for user_id in user_ids:
            # Mỗi user có 1-3 product_type yêu thích
            preferred_types = np.random.choice(
                product_types, 
                size=np.random.randint(1, min(4, len(product_types))), 
                replace=False
            )
            
            # User có xu hướng rating cao hơn cho product_type yêu thích
            user_preferences[user_id] = {
                'preferred_types': preferred_types,
                'price_sensitivity': np.random.uniform(0.3, 1.0),  # 0.3-1.0
                'brand_loyalty': np.random.uniform(0.2, 0.8),      # 0.2-0.8
                'rating_bias': np.random.normal(0, 0.3)            # bias cá nhân
            }
        
        # Tạo interactions dựa trên product characteristics
        for idx, product in df.iterrows():
            # Số lượng interactions dựa trên reviews và rating
            base_interactions = max(5, int(product['reviews'] / 15))
            rating_boost = (product['rating'] - 2.5) * 2  # rating cao -> nhiều interactions hơn
            popularity = max(1, int(base_interactions + rating_boost))
            
            # Variation ngẫu nhiên
            n_interactions = max(3, int(popularity * np.random.uniform(0.6, 1.4)))
            n_interactions = min(n_interactions, len(user_ids))
            
            # Chọn users có xu hướng thích product này
            product_type = product['product_type']
            product_price = product['price']
            product_brand = product['brand']
            
            # Users có preferred_type match có xác suất cao hơn
            potential_users = []
            for user_id in user_ids:
                prefs = user_preferences[user_id]
                prob = 0.1  # base probability
                
                if product_type in prefs['preferred_types']:
                    prob += 0.6  # tăng 60% nếu match type
                
                # Price sensitivity
                if product_price < 200:  # giá rẻ
                    prob += (1 - prefs['price_sensitivity']) * 0.3
                elif product_price > 400:  # giá cao
                    prob += prefs['price_sensitivity'] * 0.2
                
                potential_users.append((user_id, prob))
            
            # Sample users theo probability
            users_probs = [p[1] for p in potential_users]
            users_probs = np.array(users_probs) / sum(users_probs)  # normalize
            
            selected_users = np.random.choice(
                [p[0] for p in potential_users],
                size=n_interactions,
                replace=False,
                p=users_probs
            )
            
            # Tạo ratings cho selected users
            for user_id in selected_users:
                prefs = user_preferences[user_id]
                base_rating = product['rating']
                
                # Rating dựa trên user preferences
                rating_adjustment = 0
                
                if product_type in prefs['preferred_types']:
                    rating_adjustment += np.random.normal(0.5, 0.3)  # boost cho type yêu thích
                
                # Brand loyalty (nếu user đã rate brand này cao trước đó)
                rating_adjustment += prefs['rating_bias']
                
                # Thêm noise
                noise = np.random.normal(0, 0.4)
                final_rating = base_rating + rating_adjustment + noise
                
                # Clamp rating trong khoảng 1-5
                final_rating = max(1.0, min(5.0, final_rating))
                final_rating = round(final_rating, 1)
                
                user_product_data.append({
                    'user_id': user_id,
                    'product_id': idx + 1,
                    'product_name': product['product_name'],
                    'brand': product['brand'],
                    'product_type': product['product_type'],
                    'rating': final_rating,
                    'reviews': product['reviews'],
                    'sales_volume': product['sales_volume'],
                    'price': product['price']
                })
        
        # Tạo thêm cross-category interactions (ít hơn)
        n_cross = int(len(user_product_data) * 0.15)  # 15% cross interactions
        
        for _ in range(n_cross):
            user_id = np.random.choice(user_ids)
            product_idx = np.random.randint(0, len(df))
            product = df.iloc[product_idx]
            
            # Cross-category rating có xu hướng trung bình hơn
            rating = np.random.normal(3.0, 1.0)
            rating = max(1.0, min(5.0, rating))
            rating = round(rating, 1)
            
            user_product_data.append({
                'user_id': user_id,
                'product_id': product_idx + 1,
                'product_name': product['product_name'],
                'brand': product['brand'],
                'product_type': product['product_type'],
                'rating': rating,
                'reviews': product['reviews'],
                'sales_volume': product['sales_volume'],
                'price': product['price']
            })
        
        self.ratings_df = pd.DataFrame(user_product_data)
        
        # Remove duplicates (same user-product pairs)
        self.ratings_df = self.ratings_df.drop_duplicates(subset=['user_id', 'product_id'], keep='first')
        
        # Statistics
        print(f"Generated {len(self.ratings_df)} unique interactions")
        print(f"Unique users: {self.ratings_df['user_id'].nunique()}")
        print(f"Unique products: {self.ratings_df['product_id'].nunique()}")
        print(f"Average rating: {self.ratings_df['rating'].mean():.2f}")
        print(f"Rating std: {self.ratings_df['rating'].std():.2f}")
        
        # Per-user và per-product stats
        user_stats = self.ratings_df.groupby('user_id').size()
        product_stats = self.ratings_df.groupby('product_id').size()
        
        print(f"Interactions per user: avg={user_stats.mean():.1f}, min={user_stats.min()}, max={user_stats.max()}")
        print(f"Ratings per product: avg={product_stats.mean():.1f}, min={product_stats.min()}, max={product_stats.max()}")
        
        self.exporter.export_step_data(self.ratings_df, "03_user_product_matrix")
        return self.ratings_df