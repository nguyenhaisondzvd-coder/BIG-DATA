import json
from datetime import datetime

class Utils:
    @staticmethod
    def generate_report(metrics, business_analysis, ratings_df, predictions):
        """Generate comprehensive report"""
        print("\n" + "="*50)
        print("RECOMMENDATION SYSTEM EVALUATION REPORT")
        print("="*50)
        
        print(f"\nMODEL PERFORMANCE:")
        print(f"- RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        print(f"- Precision@10: {metrics.get('precision_at_10', 'N/A'):.4f}")
        
        print(f"\nBUSINESS IMPACT:")
        print(f"- Potential Revenue: ${business_analysis['potential_revenue']:,.2f}")
        print(f"- Brand Diversity: {business_analysis['brand_diversity']} brands")
        print(f"- Product Diversity: {business_analysis['product_diversity']} products")
        print(f"- User Coverage: {business_analysis['user_coverage']:.1f}%")
        
        print(f"\nDATA STATISTICS:")
        print(f"- Total Users: {ratings_df['user_id'].nunique()}")
        print(f"- Total Products: {ratings_df['product_id'].nunique()}")
        print(f"- Total Interactions: {len(ratings_df)}")
        print(f"- Average Rating: {ratings_df['rating'].mean():.2f}")
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'business_analysis': business_analysis,
            'data_statistics': {
                'total_users': ratings_df['user_id'].nunique(),
                'total_products': ratings_df['product_id'].nunique(),
                'total_interactions': len(ratings_df),
                'average_rating': float(ratings_df['rating'].mean())
            }
        }
        
        return report_data
    
    @staticmethod
    def save_report(report_data, file_path):
        """Save report to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"Report saved to {file_path}")

import pandas as pd
import os

class ExcelExporter:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_sample_to_excel(self, data, filename, sheet_name="Data", sample_size=80):
        """Export sample data (80 rows) to Excel"""
        print(f"Exporting {min(sample_size, len(data))} sample rows to {filename}...")
        
        filepath = os.path.join(self.output_dir, filename)
        
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data)
        
        # Lấy mẫu 80 dòng đầu tiên
        sample_df = df.head(sample_size)
        sample_df.to_excel(filepath, sheet_name=sheet_name, index=False)
        
        print(f"✅ Exported {len(sample_df)} sample rows to: {filepath}")
        return filepath
    
    def export_step_data(self, data, step_name, sample_size=80):
        """Export data for a specific processing step"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{step_name}_{timestamp}.xlsx"
        return self.export_sample_to_excel(data, filename, step_name, sample_size)
    
import subprocess
import os

class HDFSManager:
    def __init__(self, hdfs_url="hdfs://namenode:9000"):
        self.hdfs_url = hdfs_url
    
    def upload_to_hdfs(self, local_path, hdfs_path):
        """Upload file to HDFS"""
        cmd = f"hdfs dfs -put -f {local_path} {self.hdfs_url}{hdfs_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Uploaded {local_path} to HDFS: {hdfs_path}")
        else:
            print(f"❌ Failed to upload: {result.stderr}")
    
    def download_from_hdfs(self, hdfs_path, local_path):
        """Download file from HDFS"""
        cmd = f"hdfs dfs -get {self.hdfs_url}{hdfs_path} {local_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Downloaded from HDFS to {local_path}")
        else:
            print(f"❌ Failed to download: {result.stderr}")
    
    def list_hdfs(self, hdfs_path):
        """List files in HDFS directory"""
        cmd = f"hdfs dfs -ls {self.hdfs_url}{hdfs_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error: {result.stderr}"
    
    def setup_hdfs_directories(self):
        """Create necessary directories in HDFS"""
        directories = [
            '/user/recommendation',
            '/user/recommendation/data',
            '/user/recommendation/models', 
            '/user/recommendation/results',
            '/spark-logs'
        ]
        
        for directory in directories:
            cmd = f"hdfs dfs -mkdir -p {self.hdfs_url}{directory}"
            subprocess.run(cmd, shell=True)
        print("✅ HDFS directories created")