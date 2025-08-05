#!/usr/bin/env python3
"""
Mini-Batch K-Means Clustering for Color Data
=============================================

This script performs Mini-Batch K-means clustering on painting color features:
1. Drops non-feature columns (filename, image, page, color, index)
2. Uses 18-D color features (BGR, HSV, LAB means and std) with equal weights
3. Scales features using StandardScaler
4. Applies PCA for dimensionality reduction (~95% variance)
5. Validates k=30 using elbow method and silhouette analysis
6. Trains Mini-Batch K-Means with optimized parameters
7. Evaluates cluster quality and extracts artifacts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE
import umap.umap_ as umap
from kneed import KneeLocator
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data(file_path):
    """Load data and prepare features with equal weighting."""
    print("Loading and preparing data...")
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} samples with {df.shape[1]} columns")
    
    # Columns to drop
    columns_to_drop = ['filename', 'image', 'page', 'color', 'index']
    
    # Drop specified columns (only if they exist)
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df_features = df.drop(existing_cols_to_drop, axis=1)
        print(f"Dropped columns: {existing_cols_to_drop}")
    else:
        df_features = df.copy()
        print("No columns to drop found")
    
    print(f"After dropping columns: {df_features.shape[1]} feature columns")
    print(f"Remaining columns: {list(df_features.columns)}")
    
    # All features are treated equally (no weighting)
    X_features = df_features.copy()
    print(f"Using {X_features.shape[1]} color features with equal weights")
    
    return X_features, df

def scale_features(X):
    """Scale features to mean=0, variance=1."""
    print("\nScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Scaled features shape: {X_scaled.shape}")
    print(f"Mean after scaling: {np.mean(X_scaled):.6f}")
    print(f"Std after scaling: {np.std(X_scaled):.6f}")
    
    return X_scaled, scaler

def apply_pca(X_scaled, variance_threshold=0.95):
    """Apply PCA to reduce dimensionality while keeping 95% variance."""
    print(f"\nApplying PCA (target variance: {variance_threshold:.1%})...")
    
    # Fit PCA with all components first to see variance explained
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # Find number of components for target variance
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumsum_var >= variance_threshold) + 1
    
    print(f"Components needed for {variance_threshold:.1%} variance: {n_components}")
    print(f"Actual variance explained: {cumsum_var[n_components-1]:.3f}")
    
    # Fit PCA with optimal number of components
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"PCA output shape: {X_pca.shape}")
    
    # Plot variance explained
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
             pca_full.explained_variance_ratio_, 'bo-', alpha=0.7)
    plt.axvline(x=n_components, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Individual Variance Explained')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'ro-', alpha=0.7)
    plt.axhline(y=variance_threshold, color='g', linestyle='--', alpha=0.7)
    plt.axvline(x=n_components, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Explained')
    plt.title('Cumulative Variance Explained')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_variance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return X_pca, pca

def validate_k_value(X_pca, k_range=(20, 40), target_k=30):
    """Validate k=30 using elbow method and silhouette analysis."""
    print(f"\nValidating k values from {k_range[0]} to {k_range[1]}...")
    
    k_values = range(k_range[0], k_range[1] + 1)
    inertias = []
    silhouette_scores = []
    
    # Test different k values
    for k in k_values:
        print(f"Testing k={k}...", end=' ')
        
        # Fit Mini-Batch K-Means
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            init='k-means++',
            batch_size=1024,
            n_init=30,  # Reduced for faster validation
            max_iter=1000,  # Reduced for faster validation
            random_state=42
        )
        kmeans.fit(X_pca)
        
        # Calculate metrics
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(X_pca, kmeans.labels_)
        silhouette_scores.append(sil_score)
        
        print(f"Inertia: {kmeans.inertia_:.1f}, Silhouette: {sil_score:.3f}")
    
    # Find elbow using kneed
    try:
        knee_locator = KneeLocator(k_values, inertias, curve='convex', direction='decreasing')
        elbow_k = knee_locator.elbow
    except:
        elbow_k = None
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow plot
    ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    if elbow_k:
        ax1.axvline(x=elbow_k, color='r', linestyle='--', alpha=0.7, label=f'Elbow at k={elbow_k}')
    ax1.axvline(x=target_k, color='g', linestyle='--', alpha=0.7, label=f'Target k={target_k}')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Silhouette plot
    ax2.plot(k_values, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    ax2.axhline(y=0.4, color='orange', linestyle='--', alpha=0.7, label='Threshold (0.4)')
    ax2.axvline(x=target_k, color='g', linestyle='--', alpha=0.7, label=f'Target k={target_k}')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Average Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('k_validation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Check if k=30 is valid
    target_idx = target_k - k_range[0]
    target_silhouette = silhouette_scores[target_idx]
    
    print(f"\n=== K Validation Results ===")
    print(f"Elbow detected at k={elbow_k if elbow_k else 'Not clear'}")
    print(f"Target k={target_k}: Silhouette score = {target_silhouette:.3f}")
    print(f"Silhouette threshold (≥0.4): {'✓ PASS' if target_silhouette >= 0.4 else '✗ FAIL'}")
    
    # Find best k based on silhouette score
    best_k_idx = np.argmax(silhouette_scores)
    best_k = k_values[best_k_idx]
    best_silhouette = silhouette_scores[best_k_idx]
    
    print(f"Best k by silhouette: k={best_k} (score: {best_silhouette:.3f})")
    
    # Decision logic
    if target_silhouette >= 0.4:
        final_k = target_k
        print(f"✓ Using target k={target_k} (passes validation)")
    else:
        final_k = best_k
        print(f"✗ Target k={target_k} failed validation, using best k={best_k}")
    
    return final_k, silhouette_scores, inertias

def train_mini_batch_kmeans(X_pca, k, max_n_init=30):
    """Train Mini-Batch K-Means with monitoring."""
    print(f"\n=== Training Mini-Batch K-Means (k={k}) ===")
    
    best_inertia = float('inf')
    best_model = None
    inertias = []
    
    for i in range(max_n_init):
        print(f"Training run {i+1}/{max_n_init}...", end=' ')
        
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            init='k-means++',
            batch_size=1024,
            n_init=1,  # Single run per iteration
            max_iter=1000,
            random_state=42 + i  # Different seed each run
        )
        kmeans.fit(X_pca)
        
        inertias.append(kmeans.inertia_)
        print(f"Inertia: {kmeans.inertia_:.1f}")
        
        # Keep best model
        if kmeans.inertia_ < best_inertia:
            best_inertia = kmeans.inertia_
            best_model = kmeans
        
        # Early stopping: if improvement < 1%
        if i >= 5:  # Start checking after 5 runs
            recent_inertias = inertias[-5:]
            if (max(recent_inertias) - min(recent_inertias)) / min(recent_inertias) < 0.01:
                print(f"Early stopping at run {i+1} (improvement < 1%)")
                break
    
    print(f"Best inertia: {best_inertia:.1f}")
    return best_model

def evaluate_cluster_quality(X_pca, kmeans, min_cluster_size_pct=1.0):
    """Evaluate cluster quality with multiple metrics."""
    print("\n=== Cluster Quality Evaluation ===")
    
    labels = kmeans.labels_
    n_samples = len(labels)
    
    # 1. Cluster size analysis
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique_labels, counts))
    
    print("Cluster size distribution:")
    for cluster_id, size in cluster_sizes.items():
        pct = (size / n_samples) * 100
        status = "✓" if pct >= min_cluster_size_pct else "⚠"
        print(f"  Cluster {cluster_id}: {size:4d} samples ({pct:5.1f}%) {status}")
    
    # Check for small clusters
    small_clusters = [cid for cid, size in cluster_sizes.items() 
                     if (size / n_samples) * 100 < min_cluster_size_pct]
    
    if small_clusters:
        print(f"⚠ Warning: {len(small_clusters)} clusters below {min_cluster_size_pct}% threshold")
    else:
        print("✓ All clusters meet size threshold")
    
    # 2. Silhouette analysis
    overall_silhouette = silhouette_score(X_pca, labels)
    sample_silhouettes = silhouette_samples(X_pca, labels)
    
    print(f"\nSilhouette Analysis:")
    print(f"Overall silhouette score: {overall_silhouette:.3f}")
    
    # Per-cluster silhouette
    problematic_clusters = []
    for cluster_id in unique_labels:
        cluster_mask = labels == cluster_id
        cluster_silhouettes = sample_silhouettes[cluster_mask]
        median_sil = np.median(cluster_silhouettes)
        
        if median_sil < 0:
            problematic_clusters.append(cluster_id)
            print(f"  Cluster {cluster_id}: median={median_sil:.3f} ⚠")
        else:
            print(f"  Cluster {cluster_id}: median={median_sil:.3f} ✓")
    
    # 3. Visualizations
    create_cluster_visualizations(X_pca, labels, sample_silhouettes, cluster_sizes)
    
    return {
        'overall_silhouette': overall_silhouette,
        'cluster_sizes': cluster_sizes,
        'small_clusters': small_clusters,
        'problematic_clusters': problematic_clusters
    }

def create_cluster_visualizations(X_pca, labels, sample_silhouettes, cluster_sizes):
    """Create visualizations for cluster analysis."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Cluster size histogram
    ax1 = plt.subplot(2, 3, 1)
    sizes = list(cluster_sizes.values())
    plt.bar(range(len(sizes)), sizes, alpha=0.7)
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Samples')
    plt.title('Cluster Size Distribution')
    plt.grid(True, alpha=0.3)
    
    # 2. Silhouette plot
    ax2 = plt.subplot(2, 3, 2)
    y_lower = 0
    colors = plt.cm.tab20(np.linspace(0, 1, len(np.unique(labels))))
    
    for i, cluster_id in enumerate(sorted(np.unique(labels))):
        cluster_silhouettes = sample_silhouettes[labels == cluster_id]
        cluster_silhouettes.sort()
        
        size = len(cluster_silhouettes)
        y_upper = y_lower + size
        
        ax2.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouettes,
                         facecolor=colors[i], alpha=0.7)
        
        # Label clusters
        ax2.text(-0.05, y_lower + 0.5 * size, str(cluster_id))
        y_lower = y_upper + 10
    
    ax2.axvline(x=silhouette_score(X_pca, labels), color="red", linestyle="--")
    ax2.set_xlabel('Silhouette Score')
    ax2.set_ylabel('Cluster ID')
    ax2.set_title('Silhouette Plot')
    
    # 3. 2D projection using UMAP
    print("Creating UMAP projection...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_2d = reducer.fit_transform(X_pca)
    
    ax3 = plt.subplot(2, 3, 3)
    scatter = ax3.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab20', alpha=0.6, s=10)
    ax3.set_xlabel('UMAP 1')
    ax3.set_ylabel('UMAP 2')
    ax3.set_title('UMAP Projection Colored by Cluster')
    plt.colorbar(scatter, ax=ax3)
    
    # 4. Cluster size percentage pie chart
    ax4 = plt.subplot(2, 3, 4)
    sizes_pct = [(size/sum(cluster_sizes.values()))*100 for size in cluster_sizes.values()]
    wedges, texts, autotexts = ax4.pie(sizes_pct, labels=cluster_sizes.keys(), autopct='%1.1f%%')
    ax4.set_title('Cluster Size Distribution (%)')
    
    # 5. PCA components plot (first 2)
    ax5 = plt.subplot(2, 3, 5)
    scatter = ax5.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab20', alpha=0.6, s=10)
    ax5.set_xlabel('PC1')
    ax5.set_ylabel('PC2')
    ax5.set_title('First 2 PCA Components')
    plt.colorbar(scatter, ax=ax5)
    
    # 6. Silhouette score distribution
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(sample_silhouettes, bins=50, alpha=0.7, edgecolor='black')
    ax6.axvline(x=np.mean(sample_silhouettes), color='red', linestyle='--', 
                label=f'Mean: {np.mean(sample_silhouettes):.3f}')
    ax6.set_xlabel('Silhouette Score')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Silhouette Score Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cluster_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def extract_and_store_artifacts(kmeans, pca, scaler, X_features, original_df, output_file):
    """Extract centroids and add cluster labels to original data."""
    print("\n=== Extracting and Storing Artifacts ===")
    
    # Get cluster labels
    labels = kmeans.labels_
    
    # Add cluster labels to original dataframe
    df_with_clusters = original_df.copy()
    df_with_clusters['cluster_id'] = labels
    
    # Save updated CSV
    df_with_clusters.to_csv(output_file, index=False)
    print(f"✓ Saved data with cluster labels to: {output_file}")
    
    # Extract centroids in PCA space
    centroids_pca = kmeans.cluster_centers_
    print(f"Centroids in PCA space: {centroids_pca.shape}")
    
    # Transform centroids back to original feature space
    # Note: This gives us the centroids in the scaled feature space
    centroids_scaled = pca.inverse_transform(centroids_pca)
    
    # Unscale to get centroids in original feature space
    centroids_original = scaler.inverse_transform(centroids_scaled)
    
    # Create color feature summary for each cluster
    # Since we have 18 color features (BGR, HSV, LAB mean and std), we can create
    # a summary of the dominant color characteristics
    feature_names = ['b_bgr_mean', 'g_bgr_mean', 'r_bgr_mean', 
                    'h_hsv_mean', 's_hsv_mean', 'v_hsv_mean',
                    'l_lab_mean', 'a_lab_mean', 'b_lab_mean',
                    'b_bgr_std', 'g_bgr_std', 'r_bgr_std',
                    'h_hsv_std', 's_hsv_std', 'v_hsv_std',
                    'l_lab_std', 'a_lab_std', 'b_lab_std']
    
    # Create color profile for each cluster based on BGR means (approximate color representation)
    color_profiles = {}
    for i, centroid in enumerate(centroids_original):
        # Extract BGR mean values (first 3 features)
        b_mean, g_mean, r_mean = centroid[0], centroid[1], centroid[2]
        
        # Also extract HSV and LAB characteristics
        h_mean, s_mean, v_mean = centroid[3], centroid[4], centroid[5]
        l_mean, a_mean, b_lab_mean = centroid[6], centroid[7], centroid[8]
        
        color_profiles[f'cluster_{i}'] = {
            'bgr_mean': [b_mean, g_mean, r_mean],
            'hsv_mean': [h_mean, s_mean, v_mean],
            'lab_mean': [l_mean, a_mean, b_lab_mean],
            'dominant_color_desc': f"B:{b_mean:.2f}, G:{g_mean:.2f}, R:{r_mean:.2f}"
        }
    
    # Save centroids
    centroids_df = pd.DataFrame(centroids_original, 
                               columns=feature_names,
                               index=[f'cluster_{i}' for i in range(len(centroids_original))])
    centroids_df.to_csv('cluster_centroids.csv')
    print("✓ Saved cluster centroids to: cluster_centroids.csv")
    
    # Save color profiles
    profiles_df = pd.DataFrame(color_profiles).T
    profiles_df.to_csv('cluster_color_palettes.csv')
    print("✓ Saved color profiles to: cluster_color_palettes.csv")
    
    # Print summary statistics
    print(f"\n=== Final Results Summary ===")
    print(f"Total samples clustered: {len(labels):,}")
    print(f"Number of clusters: {len(np.unique(labels))}")
    print(f"Cluster size range: {np.min(np.bincount(labels))} - {np.max(np.bincount(labels))}")
    print(f"Final inertia: {kmeans.inertia_:.1f}")
    
    return {
        'centroids': centroids_original,
        'color_profiles': color_profiles,
        'labels': labels,
        'df_with_clusters': df_with_clusters
    }

def main():
    """Main execution function."""
    print("=== Mini-Batch K-Means Clustering for Color Data ===\n")
    
    # Configuration
    INPUT_FILE = 'data/processed/pictures.csv'
    OUTPUT_FILE = 'colour_data_with_clusters_3.0.csv'
    TARGET_K = 30
    
    try:
        # Step 1: Load and prepare data
        X_features, original_df = load_and_prepare_data(INPUT_FILE)
        
        # Step 2: Scale features
        X_scaled, scaler = scale_features(X_features)
        
        # Step 3: Apply PCA
        X_pca, pca = apply_pca(X_scaled, variance_threshold=0.95)
        
        # Step 4: Validate k value
        final_k, silhouette_scores, inertias = validate_k_value(X_pca, k_range=(20, 40), target_k=TARGET_K)
        
        # Step 5: Train Mini-Batch K-Means
        kmeans = train_mini_batch_kmeans(X_pca, final_k)
        
        # Step 6: Evaluate cluster quality
        quality_metrics = evaluate_cluster_quality(X_pca, kmeans)
        
        # Step 7: Extract and store artifacts
        artifacts = extract_and_store_artifacts(kmeans, pca, scaler, X_features, 
                                              original_df, OUTPUT_FILE)
        
        print("\n🎉 Clustering pipeline completed successfully!")
        print(f"📁 Output files created:")
        print(f"   • {OUTPUT_FILE}")
        print(f"   • cluster_centroids.csv")
        print(f"   • cluster_color_palettes.csv")
        print(f"   • pca_variance_analysis.png")
        print(f"   • k_validation_analysis.png")
        print(f"   • cluster_quality_analysis.png")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()