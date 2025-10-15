import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from modules.utils import ExcelExporter
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self, spark_session):
        self.spark = spark_session
        self.df = None
        self.ratings_df = None
        self.exporter = ExcelExporter()
    
    def load_raw_data(self, file_path):
        """Load raw data from Excel file"""
        print("Loading raw data...")
        self.df = pd.read_excel(file_path)
        print(f"Dataset shape: {self.df.shape}")
        self.exporter.export_step_data(self.df, "01_raw_data")
        return self.df
    
    def save_processed_data(self, file_path):
        """Save processed data to CSV"""
        if self.df is not None:
            self.df.to_csv(file_path, index=False)
            print(f"Processed data saved to {file_path}")
    
    def save_ratings_data(self, file_path):
        """Save ratings data to CSV"""
        if self.ratings_df is not None:
            self.ratings_df.to_csv(file_path, index=False)
            print(f"Ratings data saved to {file_path}")
    
    def load_processed_data(self, file_path):
        """Load previously processed data"""
        self.df = pd.read_csv(file_path)
        print(f"Loaded processed data: {self.df.shape}")
        return self.df
    
    def load_ratings_data(self, file_path):
        """Load previously generated ratings data"""
        self.ratings_df = pd.read_csv(file_path)
        print(f"Loaded ratings data: {self.ratings_df.shape}")
        return self.ratings_df