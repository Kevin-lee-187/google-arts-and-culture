#!/usr/bin/env python3
"""
Color Combination Generator
==========================

This script generates 30 color combinations based on cluster analysis:
1. Each combination contains 4 dominant colors (clusters)
2. Every cluster must be used with equal frequency across all combinations
3. Minimizes variance in total cluster sizes per combination
4. Uses cluster sizes and dominant color descriptions from clustering results
"""

import pandas as pd
import numpy as np
import itertools
from collections import defaultdict
import random
import json

def load_cluster_data():
    """Load cluster sizes and color descriptions."""
    print("Loading cluster data...")
    
    # Load cluster data to get sizes
    df_clusters = pd.read_csv('colour_data_with_clusters_3.0.csv')
    cluster_sizes = df_clusters['cluster_id'].value_counts().sort_index().to_dict()
    
    # Load color descriptions
    df_colors = pd.read_csv('cluster_color_palettes.csv', index_col=0)
    color_descriptions = df_colors['dominant_color_desc'].to_dict()
    
    print(f"Found {len(cluster_sizes)} clusters")
    print(f"Total samples: {sum(cluster_sizes.values())}")
    
    # Display cluster information
    print("\nCluster information:")
    for cluster_id in sorted(cluster_sizes.keys()):
        size = cluster_sizes[cluster_id]
        desc = color_descriptions[f'cluster_{cluster_id}']
        print(f"  Cluster {cluster_id:2d}: {size:4d} samples - {desc}")
    
    return cluster_sizes, color_descriptions

def calculate_optimal_frequencies(cluster_sizes, total_combinations=30, colors_per_combination=4):
    """Calculate optimal frequency for each cluster to minimize variance."""
    total_slots = total_combinations * colors_per_combination
    num_clusters = len(cluster_sizes)
    
    print(f"\nCalculating optimal frequencies:")
    print(f"Total combinations: {total_combinations}")
    print(f"Colors per combination: {colors_per_combination}")
    print(f"Total slots: {total_slots}")
    print(f"Clusters: {num_clusters}")
    print(f"Ideal frequency per cluster: {total_slots / num_clusters:.2f}")
    
    # Each cluster should appear either floor or ceil times
    base_frequency = total_slots // num_clusters
    extra_slots = total_slots % num_clusters
    
    print(f"Base frequency: {base_frequency}")
    print(f"Extra slots to distribute: {extra_slots}")
    
    # Sort clusters by size (descending) to assign frequencies optimally
    # Larger clusters get base_frequency, smaller clusters get base_frequency + 1
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    
    frequencies = {}
    large_clusters = sorted_clusters[:num_clusters - extra_slots]
    small_clusters = sorted_clusters[num_clusters - extra_slots:]
    
    # Assign frequencies
    for cluster_id, size in large_clusters:
        frequencies[cluster_id] = base_frequency
    
    for cluster_id, size in small_clusters:
        frequencies[cluster_id] = base_frequency + 1
    
    print(f"\nFrequency assignment:")
    print(f"  {len(large_clusters)} clusters appear {base_frequency} times each")
    print(f"  {len(small_clusters)} clusters appear {base_frequency + 1} times each")
    
    # Verify total
    total_check = sum(frequencies.values())
    assert total_check == total_slots, f"Frequency total mismatch: {total_check} != {total_slots}"
    
    return frequencies

def generate_combinations_optimized(cluster_sizes, frequencies, total_combinations=30):
    """Generate combinations using optimization to minimize variance."""
    print(f"\nGenerating {total_combinations} combinations using optimization approach...")
    
    # Create pool of clusters based on frequencies
    cluster_pool = []
    for cluster_id, freq in frequencies.items():
        cluster_pool.extend([cluster_id] * freq)
    
    best_combinations = None
    best_variance = float('inf')
    
    # Try multiple random starts
    for attempt in range(100):
        if attempt % 20 == 0:
            print(f"  Optimization attempt {attempt + 1}/100...")
        
        # Randomly shuffle and partition into combinations
        temp_pool = cluster_pool.copy()
        random.shuffle(temp_pool)
        
        combinations = []
        for i in range(0, len(temp_pool), 4):
            if i + 3 < len(temp_pool):
                combinations.append(tuple(temp_pool[i:i+4]))
        
        # Only consider if we have exactly the right number of combinations
        if len(combinations) == total_combinations:
            variance = calculate_combination_variance(combinations, cluster_sizes)
            
            if variance < best_variance:
                best_variance = variance
                best_combinations = combinations.copy()
                print(f"    New best variance: {variance:.1f}")
    
    # Apply local optimization using hill climbing
    if best_combinations:
        print(f"  Applying local optimization...")
        best_combinations = local_optimize(best_combinations, cluster_sizes, max_iterations=1000)
        final_variance = calculate_combination_variance(best_combinations, cluster_sizes)
        print(f"  Final variance after optimization: {final_variance:.1f}")
    
    # Display final combinations
    for i, combo in enumerate(best_combinations):
        combo_size = sum(cluster_sizes[cid] for cid in combo)
        print(f"  Combination {i+1:2d}: {combo} (size: {combo_size})")
    
    return best_combinations

def local_optimize(combinations, cluster_sizes, max_iterations=1000):
    """Apply local optimization using hill climbing."""
    current_combinations = [list(combo) for combo in combinations]
    current_variance = calculate_combination_variance(current_combinations, cluster_sizes)
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try swapping clusters between combinations
        for i in range(len(current_combinations)):
            for j in range(i + 1, len(current_combinations)):
                for pos_i in range(4):
                    for pos_j in range(4):
                        # Swap clusters
                        temp_combinations = [combo.copy() for combo in current_combinations]
                        temp_combinations[i][pos_i], temp_combinations[j][pos_j] = \
                            temp_combinations[j][pos_j], temp_combinations[i][pos_i]
                        
                        # Check if this improves variance
                        new_variance = calculate_combination_variance(temp_combinations, cluster_sizes)
                        if new_variance < current_variance:
                            current_combinations = temp_combinations
                            current_variance = new_variance
                            improved = True
                            if iteration % 100 == 0:
                                print(f"    Iteration {iteration}: variance = {current_variance:.1f}")
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    return [tuple(combo) for combo in current_combinations]

def calculate_combination_variance(combinations, cluster_sizes):
    """Calculate variance in total cluster sizes across combinations."""
    combination_sums = []
    for combo in combinations:
        total_size = sum(cluster_sizes[cluster_id] for cluster_id in combo)
        combination_sums.append(total_size)
    
    if len(combination_sums) <= 1:
        return 0
    
    return np.var(combination_sums)

def analyze_combinations(combinations, cluster_sizes, color_descriptions):
    """Analyze the generated combinations."""
    print(f"\n=== Combination Analysis ===")
    
    # Calculate combination sizes and statistics
    combination_sizes = []
    combination_details = []
    
    for i, combo in enumerate(combinations):
        total_size = sum(cluster_sizes[cluster_id] for cluster_id in combo)
        combination_sizes.append(total_size)
        
        colors = []
        for cluster_id in combo:
            desc = color_descriptions[f'cluster_{cluster_id}']
            colors.append(f"C{cluster_id}({desc})")
        
        combination_details.append({
            'combination_id': i + 1,
            'clusters': list(combo),
            'total_size': total_size,
            'colors': colors
        })
    
    # Statistics
    mean_size = np.mean(combination_sizes)
    std_size = np.std(combination_sizes)
    variance = np.var(combination_sizes)
    
    print(f"Combination size statistics:")
    print(f"  Mean: {mean_size:.1f}")
    print(f"  Std Dev: {std_size:.1f}")  
    print(f"  Variance: {variance:.1f}")
    print(f"  Min: {min(combination_sizes)}")
    print(f"  Max: {max(combination_sizes)}")
    print(f"  Range: {max(combination_sizes) - min(combination_sizes)}")
    
    # Frequency check
    frequency_count = defaultdict(int)
    for combo in combinations:
        for cluster_id in combo:
            frequency_count[cluster_id] += 1
    
    print(f"\nFrequency verification:")
    expected_freq = (len(combinations) * 4) / len(cluster_sizes)
    for cluster_id in sorted(frequency_count.keys()):
        actual = frequency_count[cluster_id]
        expected = expected_freq
        status = "✓" if abs(actual - expected) <= 1 else "✗"
        print(f"  Cluster {cluster_id:2d}: {actual} times (expected: {expected:.1f}) {status}")
    
    return combination_details, {
        'mean_size': mean_size,
        'std_size': std_size,
        'variance': variance,
        'min_size': min(combination_sizes),
        'max_size': max(combination_sizes)
    }

def save_results(combination_details, stats, color_descriptions):
    """Save results to files."""
    print(f"\n=== Saving Results ===")
    
    # Create detailed results
    results = {
        'metadata': {
            'total_combinations': len(combination_details),
            'colors_per_combination': 4,
            'statistics': stats
        },
        'combinations': combination_details
    }
    
    # Save JSON results
    with open('color_combinations_detailed.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Saved detailed results to: color_combinations_detailed.json")
    
    # Create simple CSV for easy viewing
    csv_data = []
    for combo in combination_details:
        row = {
            'combination_id': combo['combination_id'],
            'cluster_1': combo['clusters'][0],
            'cluster_2': combo['clusters'][1], 
            'cluster_3': combo['clusters'][2],
            'cluster_4': combo['clusters'][3],
            'total_size': combo['total_size'],
            'color_1': combo['colors'][0],
            'color_2': combo['colors'][1],
            'color_3': combo['colors'][2],
            'color_4': combo['colors'][3]
        }
        csv_data.append(row)
    
    df_results = pd.DataFrame(csv_data)
    df_results.to_csv('color_combinations.csv', index=False)
    print("✓ Saved CSV results to: color_combinations.csv")
    
    # Create summary report
    with open('combination_summary.txt', 'w') as f:
        f.write("Color Combination Generation Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Generated: {len(combination_details)} combinations\n")
        f.write(f"Colors per combination: 4\n")
        f.write(f"Total clusters: 21\n\n")
        f.write(f"Size Statistics:\n")
        f.write(f"  Mean size: {stats['mean_size']:.1f}\n")
        f.write(f"  Standard deviation: {stats['std_size']:.1f}\n")
        f.write(f"  Variance: {stats['variance']:.1f}\n")
        f.write(f"  Size range: {stats['min_size']} - {stats['max_size']}\n\n")
        
        f.write("Combinations:\n")
        for combo in combination_details:
            f.write(f"  {combo['combination_id']:2d}. Clusters {combo['clusters']} (size: {combo['total_size']})\n")
    
    print("✓ Saved summary to: combination_summary.txt")

def main():
    """Main execution function."""
    print("=== Color Combination Generator ===\n")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    try:
        # Step 1: Load data
        cluster_sizes, color_descriptions = load_cluster_data()
        
        # Step 2: Calculate optimal frequencies
        frequencies = calculate_optimal_frequencies(cluster_sizes)
        
        # Step 3: Generate combinations
        combinations = generate_combinations_optimized(cluster_sizes, frequencies)
        
        if len(combinations) < 30:
            print(f"⚠ Warning: Only generated {len(combinations)} combinations (target: 30)")
        
        # Step 4: Analyze results
        combination_details, stats = analyze_combinations(combinations, cluster_sizes, color_descriptions)
        
        # Step 5: Save results
        save_results(combination_details, stats, color_descriptions)
        
        print(f"\n🎉 Successfully generated {len(combinations)} color combinations!")
        print(f"📊 Variance in combination sizes: {stats['variance']:.1f}")
        print(f"📁 Output files:")
        print(f"   • color_combinations.csv")
        print(f"   • color_combinations_detailed.json") 
        print(f"   • combination_summary.txt")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()