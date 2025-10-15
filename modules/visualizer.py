import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd

class Visualizer:
    def __init__(self):
        pass
    
    def create_recommendation_plots(self, recommendations, save_path='recommendation_analysis.png'):
        """Create recommendation analysis plots"""
        print("Creating visualizations...")
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Top recommended brands
        plt.subplot(2, 2, 1)
        top_brands = recommendations['brand'].value_counts().head(10)
        top_brands.plot(kind='bar', color='skyblue')
        plt.title('Top 10 Recommended Brands')
        plt.xlabel('Brand')
        plt.ylabel('Number of Recommendations')
        plt.xticks(rotation=45)
        
        # Plot 2: Rating distribution
        plt.subplot(2, 2, 2)
        recommendations['predicted_rating'].hist(bins=20, color='lightgreen', alpha=0.7)
        plt.title('Distribution of Predicted Ratings')
        plt.xlabel('Predicted Rating')
        plt.ylabel('Frequency')
        
        # Plot 3: Price distribution
        plt.subplot(2, 2, 3)
        recommendations['price'].hist(bins=20, color='salmon', alpha=0.7)
        plt.title('Price Distribution of Recommended Products')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        
        # Plot 4: Product type distribution
        plt.subplot(2, 2, 4)
        product_types = recommendations['product_type'].value_counts().head(8)
        product_types.plot(kind='pie', autopct='%1.1f%%')
        plt.title('Product Type Distribution in Recommendations')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_recommendation_network(self, recommendations, sample_users=50):
        """Create recommendation network"""
        print(f"Creating recommendation network for {sample_users} users...")
        
        sampled_users = recommendations['user_id'].unique()[:sample_users]
        sampled_recs = recommendations[recommendations['user_id'].isin(sampled_users)]
        
        G = nx.Graph()
        
        for _, row in sampled_recs.iterrows():
            user_node = f"User_{row['user_id']}"
            product_node = f"Product_{row['product_id']}"
            
            G.add_node(user_node, type='user')
            G.add_node(product_node, type='product', 
                      product_name=row['product_name'],
                      brand=row['brand'])
            
            G.add_edge(user_node, product_node, 
                      weight=row['predicted_rating'],
                      rating=row['predicted_rating'])
        
        print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def export_network_gephi(self, G, filename="recommendation_network.gexf"):
        """Export network to GEXF format"""
        for node in G.nodes():
            if G.nodes[node]['type'] == 'user':
                G.nodes[node]['size'] = 10
                G.nodes[node]['color'] = 'blue'
            else:
                G.nodes[node]['size'] = 15
                G.nodes[node]['color'] = 'red'
        
        nx.write_gexf(G, filename)
        print(f"Network exported to {filename}")