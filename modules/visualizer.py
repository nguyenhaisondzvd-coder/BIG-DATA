import matplotlib
matplotlib.use('Agg')  # dùng backend không tương tác phù hợp cho container/headless
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import random
import os

class Visualizer:
    def __init__(self):
        pass
    
    def create_recommendation_plots(self, recommendations, save_path='results/recommendation_analysis.png'):
        """Create recommendation analysis plots"""
        print("Creating visualizations...")

        # kiểm tra dữ liệu đầu vào
        if recommendations is None or len(recommendations) == 0:
            print("⚠️ No recommendations to plot (empty DataFrame).")
            return

        # đảm bảo thư mục lưu tồn tại
        save_dir = os.path.dirname(save_path) or '.'
        os.makedirs(save_dir, exist_ok=True)

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
        # trong container headless không gọi plt.show() (sẽ không hiển thị và có thể block)
        if os.environ.get('DISPLAY'):
            plt.show()
        plt.close()
    
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
        """Export network to GEXF format with Gephi-compatible colors, sizes, and labels"""
        for node in G.nodes():
            node_type = G.nodes[node]['type']

            # 🏷️ Gán label hiển thị
            if node_type == 'user':
                label = node  # ví dụ "User_123"
                color = {'r': 66, 'g': 135, 'b': 245}  # xanh dương
                size = 12
            else:
                # Lấy thông tin sản phẩm
                product_name = G.nodes[node].get('product_name', 'Unknown Product')
                brand = G.nodes[node].get('brand', 'Unknown Brand')
                label = f"{product_name} ({brand})"
                
                # Màu random nhẹ cho sản phẩm
                color = {
                    'r': random.randint(150, 255),
                    'g': random.randint(50, 200),
                    'b': random.randint(50, 200)
                }
                size = 20

            # Gán thông tin hiển thị
            G.nodes[node]['label'] = label
            G.nodes[node]['viz'] = {'color': color, 'size': size}

        # 🎚️ Gán độ dày cho cạnh theo predicted_rating
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1.0)
            G.edges[u, v]['viz'] = {'thickness': weight}
            G.edges[u, v]['label'] = f"Rating: {weight:.2f}"

        # Xuất file theo chuẩn GEXF 1.2 (Gephi-friendly)
        nx.write_gexf(G, filename, version='1.2draft')
        print(f"✅ Network exported to {filename} (with labels & colors for Gephi)")