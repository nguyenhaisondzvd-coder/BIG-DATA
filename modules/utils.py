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