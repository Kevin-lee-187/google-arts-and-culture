#!/usr/bin/env python3
"""
Generate 73 Engineered Color Features
====================================

This script reads 12-D basic color features and generates 73 engineered features
using the exact same logic from the create_color_features_from_selection function
in recommendation_service_embedded.py.

Input: extracted_12d_color_features.csv (url + 12 basic colors)
Output: complete_85d_color_features.csv (url + 12 basic + 73 engineered = 86 total columns)
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Color names in the exact order they appear in the CSV
COLOR_NAMES = ['black', 'blue', 'brown', 'green', 'grey', 'orange', 
               'pink', 'purple', 'red', 'turquoise', 'white', 'yellow']

# Engineered feature names (73 features total) - from recommendation_service_embedded.py
ENGINEERED_FEATURES = [
    # Color ratios (26 features)
    'red_blue_ratio', 'red_green_ratio', 'red_yellow_ratio', 'red_purple_ratio',
    'blue_yellow_ratio', 'blue_orange_ratio', 'blue_green_ratio', 'blue_purple_ratio',
    'yellow_purple_ratio', 'yellow_green_ratio', 'yellow_orange_ratio',
    'green_purple_ratio', 'green_orange_ratio', 'purple_orange_ratio',
    'black_white_ratio', 'black_grey_ratio', 'black_brown_ratio',
    'white_grey_ratio', 'white_brown_ratio', 'grey_brown_ratio',
    'red_orange_ratio', 'orange_yellow_ratio', 'blue_turquoise_ratio',
    'green_turquoise_ratio', 'purple_pink_ratio', 'red_pink_ratio',
    
    # Color dominance features (26 features)
    'red_blue_dominance', 'red_green_dominance', 'red_yellow_dominance', 'red_purple_dominance',
    'blue_yellow_dominance', 'blue_orange_dominance', 'blue_green_dominance', 'blue_purple_dominance',
    'yellow_purple_dominance', 'yellow_green_dominance', 'yellow_orange_dominance',
    'green_purple_dominance', 'green_orange_dominance', 'purple_orange_dominance',
    'black_white_dominance', 'black_grey_dominance', 'black_brown_dominance',
    'white_grey_dominance', 'white_brown_dominance', 'grey_brown_dominance',
    'red_orange_dominance', 'orange_yellow_dominance', 'blue_turquoise_dominance',
    'green_turquoise_dominance', 'purple_pink_dominance', 'red_pink_dominance',
    
    # Complementary color features (3 features)
    'red_green_complementary', 'blue_orange_complementary', 'yellow_purple_complementary',
    
    # Analogous color groups (5 features)
    'analogous_group_1', 'analogous_group_2', 'analogous_group_3', 
    'analogous_group_4', 'analogous_group_5',
    
    # Color temperature and balance features (5 features)
    'warm_colors_total', 'cool_colors_total', 'neutral_colors_total',
    'warm_cool_balance', 'warm_cool_ratio',
    
    # Advanced color features (8 features)
    'num_dominant_colors', 'palette_variance', 'primary_dominance', 'color_diversity',
    'dominant_is_warm', 'dominant_is_cool', 'dominant_is_neutral', 'color_balance'
]

def create_engineered_features_from_basic_colors(basic_colors):
    """
    Create 73 engineered features from 12 basic color percentages
    Based on create_color_features_from_selection from recommendation_service_embedded.py
    
    Args:
        basic_colors: dict with keys like {'black': 0.1, 'blue': 0.2, ...}
    
    Returns:
        dict with 73 engineered features
    """
    features = {}
    
    # Extract individual color percentages
    black_pct = basic_colors.get('black', 0.0)
    blue_pct = basic_colors.get('blue', 0.0)
    brown_pct = basic_colors.get('brown', 0.0)
    green_pct = basic_colors.get('green', 0.0)
    grey_pct = basic_colors.get('grey', 0.0)
    orange_pct = basic_colors.get('orange', 0.0)
    pink_pct = basic_colors.get('pink', 0.0)
    purple_pct = basic_colors.get('purple', 0.0)
    red_pct = basic_colors.get('red', 0.0)
    turquoise_pct = basic_colors.get('turquoise', 0.0)
    white_pct = basic_colors.get('white', 0.0)
    yellow_pct = basic_colors.get('yellow', 0.0)
    
    # Color ratios (26 features)
    features['red_blue_ratio'] = red_pct / (blue_pct + 0.001)
    features['red_green_ratio'] = red_pct / (green_pct + 0.001)
    features['red_yellow_ratio'] = red_pct / (yellow_pct + 0.001)
    features['red_purple_ratio'] = red_pct / (purple_pct + 0.001)
    features['blue_yellow_ratio'] = blue_pct / (yellow_pct + 0.001)
    features['blue_orange_ratio'] = blue_pct / (orange_pct + 0.001)
    features['blue_green_ratio'] = blue_pct / (green_pct + 0.001)
    features['blue_purple_ratio'] = blue_pct / (purple_pct + 0.001)
    features['yellow_purple_ratio'] = yellow_pct / (purple_pct + 0.001)
    features['yellow_green_ratio'] = yellow_pct / (green_pct + 0.001)
    features['yellow_orange_ratio'] = yellow_pct / (orange_pct + 0.001)
    features['green_purple_ratio'] = green_pct / (purple_pct + 0.001)
    features['green_orange_ratio'] = green_pct / (orange_pct + 0.001)
    features['purple_orange_ratio'] = purple_pct / (orange_pct + 0.001)
    features['black_white_ratio'] = black_pct / (white_pct + 0.001)
    features['black_grey_ratio'] = black_pct / (grey_pct + 0.001)
    features['black_brown_ratio'] = black_pct / (brown_pct + 0.001)
    features['white_grey_ratio'] = white_pct / (grey_pct + 0.001)
    features['white_brown_ratio'] = white_pct / (brown_pct + 0.001)
    features['grey_brown_ratio'] = grey_pct / (brown_pct + 0.001)
    features['red_orange_ratio'] = red_pct / (orange_pct + 0.001)
    features['orange_yellow_ratio'] = orange_pct / (yellow_pct + 0.001)
    features['blue_turquoise_ratio'] = blue_pct / (turquoise_pct + 0.001)
    features['green_turquoise_ratio'] = green_pct / (turquoise_pct + 0.001)
    features['purple_pink_ratio'] = purple_pct / (pink_pct + 0.001)
    features['red_pink_ratio'] = red_pct / (pink_pct + 0.001)
    
    # Color dominance features (26 features)
    features['red_blue_dominance'] = 1.0 if red_pct > blue_pct else 0.0
    features['red_green_dominance'] = 1.0 if red_pct > green_pct else 0.0
    features['red_yellow_dominance'] = 1.0 if red_pct > yellow_pct else 0.0
    features['red_purple_dominance'] = 1.0 if red_pct > purple_pct else 0.0
    features['blue_yellow_dominance'] = 1.0 if blue_pct > yellow_pct else 0.0
    features['blue_orange_dominance'] = 1.0 if blue_pct > orange_pct else 0.0
    features['blue_green_dominance'] = 1.0 if blue_pct > green_pct else 0.0
    features['blue_purple_dominance'] = 1.0 if blue_pct > purple_pct else 0.0
    features['yellow_purple_dominance'] = 1.0 if yellow_pct > purple_pct else 0.0
    features['yellow_green_dominance'] = 1.0 if yellow_pct > green_pct else 0.0
    features['yellow_orange_dominance'] = 1.0 if yellow_pct > orange_pct else 0.0
    features['green_purple_dominance'] = 1.0 if green_pct > purple_pct else 0.0
    features['green_orange_dominance'] = 1.0 if green_pct > orange_pct else 0.0
    features['purple_orange_dominance'] = 1.0 if purple_pct > orange_pct else 0.0
    features['black_white_dominance'] = 1.0 if black_pct > white_pct else 0.0
    features['black_grey_dominance'] = 1.0 if black_pct > grey_pct else 0.0
    features['black_brown_dominance'] = 1.0 if black_pct > brown_pct else 0.0
    features['white_grey_dominance'] = 1.0 if white_pct > grey_pct else 0.0
    features['white_brown_dominance'] = 1.0 if white_pct > brown_pct else 0.0
    features['grey_brown_dominance'] = 1.0 if grey_pct > brown_pct else 0.0
    features['red_orange_dominance'] = 1.0 if red_pct > orange_pct else 0.0
    features['orange_yellow_dominance'] = 1.0 if orange_pct > yellow_pct else 0.0
    features['blue_turquoise_dominance'] = 1.0 if blue_pct > turquoise_pct else 0.0
    features['green_turquoise_dominance'] = 1.0 if green_pct > turquoise_pct else 0.0
    features['purple_pink_dominance'] = 1.0 if purple_pct > pink_pct else 0.0
    features['red_pink_dominance'] = 1.0 if red_pct > pink_pct else 0.0
    
    # Complementary color features (3 features)
    features['red_green_complementary'] = 1.0 if (red_pct > 0.1 and green_pct > 0.1) else 0.0
    features['blue_orange_complementary'] = 1.0 if (blue_pct > 0.1 and orange_pct > 0.1) else 0.0
    features['yellow_purple_complementary'] = 1.0 if (yellow_pct > 0.1 and purple_pct > 0.1) else 0.0
    
    # Analogous color groups (5 features)
    # Group 1: Red, Orange, Yellow
    features['analogous_group_1'] = 1.0 if (red_pct > 0.05 and orange_pct > 0.05 and yellow_pct > 0.05) else 0.0
    # Group 2: Yellow, Green, Blue
    features['analogous_group_2'] = 1.0 if (yellow_pct > 0.05 and green_pct > 0.05 and blue_pct > 0.05) else 0.0
    # Group 3: Blue, Purple, Red
    features['analogous_group_3'] = 1.0 if (blue_pct > 0.05 and purple_pct > 0.05 and red_pct > 0.05) else 0.0
    # Group 4: Green, Blue, Turquoise
    features['analogous_group_4'] = 1.0 if (green_pct > 0.05 and blue_pct > 0.05 and turquoise_pct > 0.05) else 0.0
    # Group 5: Purple, Pink, Red
    features['analogous_group_5'] = 1.0 if (purple_pct > 0.05 and pink_pct > 0.05 and red_pct > 0.05) else 0.0
    
    # Color temperature and balance features (5 features)
    warm_colors = red_pct + orange_pct + yellow_pct + brown_pct
    cool_colors = blue_pct + green_pct + purple_pct + turquoise_pct
    neutral_colors = black_pct + white_pct + grey_pct
    
    features['warm_colors_total'] = warm_colors
    features['cool_colors_total'] = cool_colors
    features['neutral_colors_total'] = neutral_colors
    features['warm_cool_balance'] = abs(warm_colors - cool_colors)
    features['warm_cool_ratio'] = warm_colors / (cool_colors + 0.001)
    
    # Advanced color features (8 features)
    # Number of dominant colors (colors with >10% presence)
    dominant_count = sum(1 for pct in basic_colors.values() if pct > 0.1)
    features['num_dominant_colors'] = dominant_count
    
    # Palette variance (standard deviation of color percentages)
    color_values = list(basic_colors.values())
    if len(color_values) > 1:
        features['palette_variance'] = np.std(color_values)
    else:
        features['palette_variance'] = 0.0
    
    # Primary dominance (check if primary colors dominate)
    primary_colors = red_pct + blue_pct + yellow_pct
    features['primary_dominance'] = primary_colors
    
    # Color diversity (number of colors with >5% presence)
    diverse_count = sum(1 for pct in basic_colors.values() if pct > 0.05)
    features['color_diversity'] = diverse_count
    
    # Dominant color type features (3 features)
    max_color = max(basic_colors.items(), key=lambda x: x[1])[0]
    features['dominant_is_warm'] = 1.0 if max_color in ['red', 'orange', 'yellow', 'brown'] else 0.0
    features['dominant_is_cool'] = 1.0 if max_color in ['blue', 'green', 'purple', 'turquoise'] else 0.0
    features['dominant_is_neutral'] = 1.0 if max_color in ['black', 'white', 'grey'] else 0.0
    
    # Color balance (inverse of palette variance for better interpretation)
    features['color_balance'] = 1.0 - features['palette_variance']
    
    return features

def process_batch(df_batch, batch_num):
    """Process a batch of rows and return results"""
    print(f"Processing batch {batch_num}: {len(df_batch)} rows")
    
    results = []
    successful = 0
    failed = 0
    
    for idx, row in df_batch.iterrows():
        try:
            # Extract basic colors from the row
            basic_colors = {color: float(row[color]) for color in COLOR_NAMES}
            
            # Generate engineered features
            engineered_features = create_engineered_features_from_basic_colors(basic_colors)
            
            # Combine URL, basic colors, and engineered features
            result = {'url': row['url']}
            result.update(basic_colors)  # Add 12 basic colors
            result.update(engineered_features)  # Add 73 engineered features
            
            results.append(result)
            successful += 1
            
            if successful % 100 == 0:
                print(f"  Processed {successful} rows...")
                
        except Exception as e:
            print(f"  Error processing row {idx}: {str(e)}")
            failed += 1
    
    print(f"  Batch {batch_num} complete: {successful} successful, {failed} failed")
    return results

def validate_features(df):
    """Validate the generated features"""
    print("\n=== Feature Validation ===")
    
    # Check basic colors (should be between 0 and 1)
    print("Basic color features (12):")
    for color in COLOR_NAMES:
        values = df[color].values
        print(f"  {color:10s}: min={np.min(values):.4f}, max={np.max(values):.4f}, mean={np.mean(values):.4f}")
    
    # Check some key engineered features
    print("\nKey engineered features:")
    key_features = ['warm_colors_total', 'cool_colors_total', 'neutral_colors_total', 
                   'palette_variance', 'primary_dominance', 'color_balance']
    
    for feature in key_features:
        if feature in df.columns:
            values = df[feature].values
            print(f"  {feature:18s}: min={np.min(values):.4f}, max={np.max(values):.4f}, mean={np.mean(values):.4f}")
    
    # Check binary features (should be 0 or 1)
    print("\nBinary feature validation:")
    binary_features = [f for f in ENGINEERED_FEATURES if 'dominance' in f or 'complementary' in f or 'analogous' in f or 'dominant_is_' in f]
    
    binary_issues = 0
    for feature in binary_features[:10]:  # Check first 10 binary features
        if feature in df.columns:
            values = df[feature].values
            unique_vals = np.unique(values)
            if not all(val in [0.0, 1.0] for val in unique_vals):
                print(f"  ⚠️  {feature}: non-binary values {unique_vals}")
                binary_issues += 1
    
    if binary_issues == 0:
        print(f"  ✅ All binary features contain only 0.0 and 1.0 values")
    
    # Overall validation
    total_features = 12 + 73  # 12 basic + 73 engineered
    actual_features = len([col for col in df.columns if col != 'url'])
    
    print(f"\n=== Overall Validation ===")
    print(f"Expected features: {total_features} (12 basic + 73 engineered)")
    print(f"Actual features: {actual_features}")
    print(f"Total columns: {len(df.columns)} (including URL)")
    
    if actual_features == total_features:
        print("✅ Feature count matches expectation!")
    else:
        print(f"⚠️  Feature count mismatch!")

def main():
    """Main function to generate engineered features"""
    print("=== Generate 73 Engineered Color Features ===\n")
    
    # Configuration
    INPUT_FILE = 'extracted_12d_color_features.csv'
    OUTPUT_FILE = 'complete_85d_color_features.csv'
    BATCH_SIZE = 500  # Process in batches to manage memory
    
    # Check input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file '{INPUT_FILE}' not found!")
        print("Please make sure you have run the batch color extraction script first.")
        return
    
    # Load input data
    print(f"Loading input data from {INPUT_FILE}...")
    try:
        df_input = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(df_input)} records with {len(df_input.columns)} columns")
        
        # Verify expected columns
        missing_colors = [color for color in COLOR_NAMES if color not in df_input.columns]
        if missing_colors:
            print(f"❌ Missing color columns: {missing_colors}")
            return
        
        print("✅ All 12 basic color columns found")
        
    except Exception as e:
        print(f"❌ Error loading input file: {e}")
        return
    
    # Process data in batches
    print(f"\nProcessing {len(df_input)} records in batches of {BATCH_SIZE}...")
    
    all_results = []
    num_batches = (len(df_input) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_num in range(num_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min((batch_num + 1) * BATCH_SIZE, len(df_input))
        
        df_batch = df_input.iloc[start_idx:end_idx]
        batch_results = process_batch(df_batch, batch_num + 1)
        all_results.extend(batch_results)
    
    # Create final DataFrame
    print(f"\nCombining results...")
    df_final = pd.DataFrame(all_results)
    
    # Ensure column order: url, basic colors, engineered features
    column_order = ['url'] + COLOR_NAMES + ENGINEERED_FEATURES
    df_final = df_final[column_order]
    
    # Save results
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Results saved to {OUTPUT_FILE}")
    print(f"   Total records: {len(df_final)}")
    print(f"   Total columns: {len(df_final.columns)} (1 URL + 12 basic + 73 engineered)")
    
    # Validate results
    validate_features(df_final)
    
    # Show sample results
    print(f"\n=== Sample Results ===")
    print("First 3 rows (URL + first 5 basic colors + first 5 engineered features):")
    sample_cols = ['url'] + COLOR_NAMES[:5] + ENGINEERED_FEATURES[:5]
    print(df_final[sample_cols].head(3).to_string(index=False))
    
    print(f"\n🎉 Successfully generated 85-D color features!")
    print(f"📁 Output: {OUTPUT_FILE}")
    print(f"📊 Records: {len(df_final):,}")
    print(f"📈 Features: 12 basic + 73 engineered = 85 total")

if __name__ == "__main__":
    main()