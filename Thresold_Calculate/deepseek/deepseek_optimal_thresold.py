import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_results_and_suggest_thresholds(results_file):
    # Load results
    df = pd.read_csv(results_file)
    
    # Separate by query type
    product_searches = df[df['type'] == 'product_search']
    non_product = df[df['type'] != 'product_search']
    
    # Calculate statistics
    stats = {
        'metric': ['top_score', 'avg_score', 'centroid_distance'],
        'product_mean': [
            product_searches['top_score'].mean(),
            product_searches['avg_score'].mean(),
            product_searches['centroid_distance'].mean()
        ],
        'product_std': [
            product_searches['top_score'].std(),
            product_searches['avg_score'].std(),
            product_searches['centroid_distance'].std()
        ],
        'non_product_mean': [
            non_product['top_score'].mean(),
            non_product['avg_score'].mean(),
            non_product['centroid_distance'].mean()
        ],
        'non_product_std': [
            non_product['top_score'].std(),
            non_product['avg_score'].std(),
            non_product['centroid_distance'].std()
        ]
    }
    
    stats_df = pd.DataFrame(stats)
    
    # Suggest thresholds (using mean - 1std for product searches as minimum)
    suggested_thresholds = {
        'top_score_thresh': max(0.1, stats_df.loc[0, 'product_mean'] - stats_df.loc[0, 'product_std']),
        'avg_score_thresh': max(0.1, stats_df.loc[1, 'product_mean'] - stats_df.loc[1, 'product_std']),
        'centroid_thresh': min(2.0, stats_df.loc[2, 'product_mean'] + stats_df.loc[2, 'product_std'])
    }
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Top Score
    plt.subplot(1, 3, 1)
    plt.hist(product_searches['top_score'], alpha=0.5, label='Product Searches')
    plt.hist(non_product['top_score'], alpha=0.5, label='Non-Product')
    plt.axvline(x=suggested_thresholds['top_score_thresh'], color='r', linestyle='--')
    plt.title('Top Score Distribution')
    plt.legend()
    
    # Avg Score
    plt.subplot(1, 3, 2)
    plt.hist(product_searches['avg_score'], alpha=0.5, label='Product Searches')
    plt.hist(non_product['avg_score'], alpha=0.5, label='Non-Product')
    plt.axvline(x=suggested_thresholds['avg_score_thresh'], color='r', linestyle='--')
    plt.title('Average Score Distribution')
    plt.legend()
    
    # Centroid Distance
    plt.subplot(1, 3, 3)
    plt.hist(product_searches['centroid_distance'], alpha=0.5, label='Product Searches')
    plt.hist(non_product['centroid_distance'], alpha=0.5, label='Non-Product')
    plt.axvline(x=suggested_thresholds['centroid_thresh'], color='r', linestyle='--')
    plt.title('Centroid Distance Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('threshold_distributions.png')
    plt.close()
    
    # Calculate effectiveness metrics
    def evaluate_thresholds(thresholds, df):
        tp = len(df[(df['type'] == 'product_search') & 
                   (df['top_score'] > thresholds['top_score_thresh']) & 
                   (df['avg_score'] > thresholds['avg_score_thresh']) & 
                   (df['centroid_distance'] < thresholds['centroid_thresh'])])
        
        fp = len(df[(df['type'] != 'product_search') & 
                   (df['top_score'] > thresholds['top_score_thresh']) & 
                   (df['avg_score'] > thresholds['avg_score_thresh']) & 
                   (df['centroid_distance'] < thresholds['centroid_thresh'])])
        
        fn = len(df[(df['type'] == 'product_search') & 
                   ~((df['top_score'] > thresholds['top_score_thresh']) & 
                    (df['avg_score'] > thresholds['avg_score_thresh']) & 
                    (df['centroid_distance'] < thresholds['centroid_thresh']))])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    current_thresholds = {'top_score_thresh': 0.4, 'avg_score_thresh': 0.3, 'centroid_thresh': 1.2}
    current_perf = evaluate_thresholds(current_thresholds, df)
    suggested_perf = evaluate_thresholds(suggested_thresholds, df)
    
    # Create comparison table
    comparison = pd.DataFrame({
        'threshold': ['current', 'suggested'],
        'top_score_thresh': [current_thresholds['top_score_thresh'], suggested_thresholds['top_score_thresh']],
        'avg_score_thresh': [current_thresholds['avg_score_thresh'], suggested_thresholds['avg_score_thresh']],
        'centroid_thresh': [current_thresholds['centroid_thresh'], suggested_thresholds['centroid_thresh']],
        'precision': [current_perf['precision'], suggested_perf['precision']],
        'recall': [current_perf['recall'], suggested_perf['recall']],
        'f1_score': [current_perf['f1_score'], suggested_perf['f1_score']]
    })
    
    # Save suggestions
    with open('threshold_suggestions.txt', 'w') as f:
        f.write("=== Current Thresholds ===\n")
        f.write(f"Top Score Threshold: {current_thresholds['top_score_thresh']}\n")
        f.write(f"Avg Score Threshold: {current_thresholds['avg_score_thresh']}\n")
        f.write(f"Centroid Threshold: {current_thresholds['centroid_thresh']}\n")
        f.write(f"Precision: {current_perf['precision']:.2f}, Recall: {current_perf['recall']:.2f}, F1: {current_perf['f1_score']:.2f}\n\n")
        
        f.write("=== Suggested Thresholds ===\n")
        f.write(f"Top Score Threshold: {suggested_thresholds['top_score_thresh']:.3f}\n")
        f.write(f"Avg Score Threshold: {suggested_thresholds['avg_score_thresh']:.3f}\n")
        f.write(f"Centroid Threshold: {suggested_thresholds['centroid_thresh']:.3f}\n")
        f.write(f"Precision: {suggested_perf['precision']:.2f}, Recall: {suggested_perf['recall']:.2f}, F1: {suggested_perf['f1_score']:.2f}\n\n")
        
        f.write("=== Recommendation ===\n")
        f.write("These thresholds attempt to maximize F1 score by balancing precision and recall.\n")
        f.write("You may want to adjust them based on whether you prioritize precision (fewer false positives)\n")
        f.write("or recall (fewer false negatives) in your application.\n")
    
    print("Analysis complete. Check threshold_suggestions.txt and threshold_distributions.png")
    return suggested_thresholds, comparison

# Run the analysis
if __name__ == "__main__":
    suggested_thresholds, comparison = analyze_results_and_suggest_thresholds('threshold_analysis_results.csv')
    print("\nSuggested Thresholds:")
    print(suggested_thresholds)
    print("\nPerformance Comparison:")
    print(comparison)