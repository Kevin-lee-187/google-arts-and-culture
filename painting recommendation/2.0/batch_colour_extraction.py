#!/usr/bin/env python3
"""
Batch Color Feature Extraction from URLs
=========================================

This script processes image URLs in batches to extract 12-D basic colour features:
1. Downloads images from URLs in batches of ~38-39 URLs each (100 total batches)
2. Extracts many raw colours from each image using k-means clustering
3. Maps raw colours to 12 basic colours using CIEDE2000 distance
4. Normalizes features to ensure values are between 0 and 1
5. Saves progress after each batch with error handling
"""

import pandas as pd
import numpy as np
import requests
from PIL import Image
from sklearn.cluster import KMeans
import math
import os
import time
import traceback
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Basic colours with RGB values (from recommendation_service_embedded.py)
BASIC_COLOURS = {
    'black': (0, 0, 0),
    'blue': (0, 0, 255),
    'brown': (139, 69, 19),
    'green': (0, 128, 0),
    'grey': (128, 128, 128),
    'orange': (255, 165, 0),
    'pink': (255, 192, 203),
    'purple': (128, 0, 128),
    'red': (255, 0, 0),
    'turquoise': (64, 224, 208),
    'white': (255, 255, 255),
    'yellow': (255, 255, 0)
}

# Color names in order for consistent feature extraction
COLOR_NAMES = ['black', 'blue', 'brown', 'green', 'grey', 'orange', 
               'pink', 'purple', 'red', 'turquoise', 'white', 'yellow']

def rgb_to_xyz(rgb):
    """Convert RGB to XYZ color space (from recommendation_service_embedded.py)"""
    r, g, b = [x/255.0 for x in rgb]
    
    # Gamma correction
    def gamma_correction(c):
        if c > 0.04045:
            return pow((c + 0.055) / 1.055, 2.4)
        else:
            return c / 12.92
    
    r = gamma_correction(r)
    g = gamma_correction(g)
    b = gamma_correction(b)
    
    # Convert to XYZ (Observer = 2°, Illuminant = D65)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    
    return (x, y, z)

def xyz_to_lab(xyz):
    """Convert XYZ to LAB color space (from recommendation_service_embedded.py)"""
    x, y, z = xyz
    
    # Normalize for D65 illuminant
    x = x / 0.95047
    y = y / 1.00000
    z = z / 1.08883
    
    def f(t):
        if t > 0.008856:
            return pow(t, 1/3)
        else:
            return (7.787 * t) + (16/116)
    
    fx = f(x)
    fy = f(y)
    fz = f(z)
    
    l = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    return (l, a, b)

def rgb_to_lab(rgb):
    """Convert RGB directly to LAB (from recommendation_service_embedded.py)"""
    xyz = rgb_to_xyz(rgb)
    return xyz_to_lab(xyz)

def ciede2000_distance(rgb1, rgb2):
    """
    Calculate CIEDE2000 color difference between two RGB colors
    (from recommendation_service_embedded.py)
    """
    # Convert to LAB
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)
    
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    # Calculate C and h
    C1 = math.sqrt(a1*a1 + b1*b1)
    C2 = math.sqrt(a2*a2 + b2*b2)
    
    # Calculate mean C
    C_mean = (C1 + C2) / 2
    
    # Calculate G
    G = 0.5 * (1 - math.sqrt(pow(C_mean, 7) / (pow(C_mean, 7) + pow(25, 7))))
    
    # Calculate a'
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)
    
    # Calculate C'
    C1_prime = math.sqrt(a1_prime*a1_prime + b1*b1)
    C2_prime = math.sqrt(a2_prime*a2_prime + b2*b2)
    
    # Calculate h'
    def calc_h_prime(a_prime, b):
        if a_prime == 0 and b == 0:
            return 0
        h = math.atan2(b, a_prime) * 180 / math.pi
        return h if h >= 0 else h + 360
    
    h1_prime = calc_h_prime(a1_prime, b1)
    h2_prime = calc_h_prime(a2_prime, b2)
    
    # Calculate delta values
    delta_L = L2 - L1
    delta_C = C2_prime - C1_prime
    
    # Calculate delta_h
    if C1_prime * C2_prime == 0:
        delta_h_prime = 0
    elif abs(h2_prime - h1_prime) <= 180:
        delta_h_prime = h2_prime - h1_prime
    elif h2_prime - h1_prime > 180:
        delta_h_prime = h2_prime - h1_prime - 360
    else:
        delta_h_prime = h2_prime - h1_prime + 360
    
    delta_H = 2 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(delta_h_prime / 2))
    
    # Calculate mean values
    L_mean = (L1 + L2) / 2
    C_mean_prime = (C1_prime + C2_prime) / 2
    
    if C1_prime * C2_prime == 0:
        h_mean_prime = h1_prime + h2_prime
    elif abs(h1_prime - h2_prime) <= 180:
        h_mean_prime = (h1_prime + h2_prime) / 2
    elif abs(h1_prime - h2_prime) > 180 and (h1_prime + h2_prime) < 360:
        h_mean_prime = (h1_prime + h2_prime + 360) / 2
    else:
        h_mean_prime = (h1_prime + h2_prime - 360) / 2
    
    # Calculate T
    T = (1 - 0.17 * math.cos(math.radians(h_mean_prime - 30)) + 
         0.24 * math.cos(math.radians(2 * h_mean_prime)) + 
         0.32 * math.cos(math.radians(3 * h_mean_prime + 6)) - 
         0.20 * math.cos(math.radians(4 * h_mean_prime - 63)))
    
    # Calculate weighting functions
    SL = 1 + ((0.015 * pow(L_mean - 50, 2)) / math.sqrt(20 + pow(L_mean - 50, 2)))
    SC = 1 + 0.045 * C_mean_prime
    SH = 1 + 0.015 * C_mean_prime * T
    
    # Calculate rotation term
    delta_theta = 30 * math.exp(-pow((h_mean_prime - 275) / 25, 2))
    RC = 2 * math.sqrt(pow(C_mean_prime, 7) / (pow(C_mean_prime, 7) + pow(25, 7)))
    RT = -RC * math.sin(2 * math.radians(delta_theta))
    
    # Calculate final CIEDE2000 difference
    kL = kC = kH = 1  # Reference conditions
    
    delta_E = math.sqrt(
        pow(delta_L / (kL * SL), 2) + 
        pow(delta_C / (kC * SC), 2) + 
        pow(delta_H / (kH * SH), 2) + 
        RT * (delta_C / (kC * SC)) * (delta_H / (kH * SH))
    )
    
    return delta_E

def download_image_from_url(url, timeout=10):
    """Download image from URL with timeout and error handling"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        
        # Load image from bytes
        image = Image.open(BytesIO(response.content))
        image = image.convert('RGB')
        return image
    except Exception as e:
        print(f"    Error downloading {url}: {str(e)}")
        return None

def extract_enhanced_colours_from_image(image, n_colours=30):
    """
    Extract many raw colours from an image using k-means clustering
    Enhanced version that extracts more colours for better mapping
    """
    try:
        # Convert image to numpy array
        image_array = np.array(image)
        
        # Reshape to get all pixels
        pixels = image_array.reshape(-1, 3)
        
        # Remove duplicate pixels and sample for efficiency
        unique_pixels = np.unique(pixels, axis=0)
        
        # If we have too many unique pixels, sample them
        if len(unique_pixels) > 10000:
            sample_indices = np.random.choice(len(unique_pixels), 10000, replace=False)
            unique_pixels = unique_pixels[sample_indices]
        
        # Adjust n_colours if we have fewer unique pixels than clusters
        actual_clusters = min(n_colours, len(unique_pixels))
        
        if actual_clusters < 2:
            return None, None
        
        # Apply k-means clustering
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        kmeans.fit(unique_pixels)
        
        # Get cluster centers (dominant colours)
        dominant_colours = kmeans.cluster_centers_.astype(int)
        
        # Get labels for all pixels to calculate percentages
        all_labels = kmeans.predict(pixels)
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        percentages = counts / len(pixels)
        
        return dominant_colours, percentages
        
    except Exception as e:
        print(f"    Error in color extraction: {str(e)}")
        return None, None

def map_raw_colours_to_basic_enhanced(raw_colours, percentages):
    """
    Map extracted raw colours to 12 basic colours using CIEDE2000 distance
    Enhanced version with better normalization
    """
    if raw_colours is None or percentages is None:
        return None
    
    # Initialize color mapping
    colour_mapping = {color: 0.0 for color in COLOR_NAMES}
    
    for i, raw_colour in enumerate(raw_colours):
        # Find the closest basic colour using CIEDE2000
        min_distance = float('inf')
        closest_basic_colour = None
        
        for basic_colour_name, basic_colour_rgb in BASIC_COLOURS.items():
            distance = ciede2000_distance(raw_colour, basic_colour_rgb)
            if distance < min_distance:
                min_distance = distance
                closest_basic_colour = basic_colour_name
        
        # Add percentage to the closest basic colour
        colour_mapping[closest_basic_colour] += percentages[i]
    
    # Normalize to ensure sum = 1 and all values > 0
    total = sum(colour_mapping.values())
    if total > 0:
        # Normalize so sum = 1
        for color in COLOR_NAMES:
            colour_mapping[color] = colour_mapping[color] / total
        
        # Ensure minimum value (smooth to prevent 0 values)
        min_value = 0.001  # Minimum 0.1%
        smoothing_factor = min_value * len(COLOR_NAMES)
        
        # Apply smoothing
        for color in COLOR_NAMES:
            colour_mapping[color] = colour_mapping[color] * (1 - smoothing_factor) + min_value
    else:
        # Fallback: equal distribution
        equal_value = 1.0 / len(COLOR_NAMES)
        for color in COLOR_NAMES:
            colour_mapping[color] = equal_value
    
    # Final verification: ensure all values are between 0 and 1
    for color in COLOR_NAMES:
        colour_mapping[color] = max(0.001, min(0.999, colour_mapping[color]))
    
    return colour_mapping

def process_single_url(url):
    """Process a single URL and extract 12-D color features"""
    try:
        # Download image
        image = download_image_from_url(url)
        if image is None:
            return None
        
        # Extract enhanced raw colours (more clusters for better coverage)
        raw_colours, percentages = extract_enhanced_colours_from_image(image, n_colours=30)
        if raw_colours is None:
            return None
        
        # Map to 12 basic colours
        basic_colour_features = map_raw_colours_to_basic_enhanced(raw_colours, percentages)
        if basic_colour_features is None:
            return None
        
        # Create feature vector in consistent order
        features = {
            'url': url,
            **{color: basic_colour_features[color] for color in COLOR_NAMES}
        }
        
        return features
        
    except Exception as e:
        print(f"    Error processing {url}: {str(e)}")
        return None

def save_batch_results(batch_results, batch_num, output_dir="batch_results"):
    """Save batch results to CSV file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    batch_df = pd.DataFrame(batch_results)
    filename = f"{output_dir}/batch_{batch_num:03d}.csv"
    batch_df.to_csv(filename, index=False)
    print(f"  Saved batch {batch_num} results to {filename}")

def load_existing_results(output_dir="batch_results"):
    """Load existing batch results to resume processing"""
    if not os.path.exists(output_dir):
        return [], set()
    
    all_results = []
    processed_urls = set()
    
    batch_files = sorted([f for f in os.listdir(output_dir) if f.startswith('batch_') and f.endswith('.csv')])
    
    for batch_file in batch_files:
        try:
            batch_df = pd.read_csv(f"{output_dir}/{batch_file}")
            all_results.extend(batch_df.to_dict('records'))
            processed_urls.update(batch_df['url'].tolist())
            print(f"  Loaded {len(batch_df)} results from {batch_file}")
        except Exception as e:
            print(f"  Error loading {batch_file}: {e}")
    
    return all_results, processed_urls

def main():
    """Main function to process URLs in batches"""
    print("=== Batch Color Feature Extraction ===\n")
    
    # Configuration
    INPUT_FILE = 'URLs.csv'
    NUM_BATCHES = 100
    OUTPUT_DIR = 'batch_results'
    FINAL_OUTPUT = 'extracted_12d_color_features.csv'
    
    # Load URLs
    print(f"Loading URLs from {INPUT_FILE}...")
    try:
        urls_df = pd.read_csv(INPUT_FILE)
        all_urls = urls_df['url'].tolist()
        print(f"Loaded {len(all_urls)} URLs")
    except Exception as e:
        print(f"Error loading URLs: {e}")
        return
    
    # Load existing results to resume processing
    print("\nChecking for existing results...")
    existing_results, processed_urls = load_existing_results(OUTPUT_DIR)
    print(f"Found {len(existing_results)} existing results, {len(processed_urls)} URLs already processed")
    
    # Filter out already processed URLs
    remaining_urls = [url for url in all_urls if url not in processed_urls]
    print(f"Remaining URLs to process: {len(remaining_urls)}")
    
    if len(remaining_urls) == 0:
        print("All URLs already processed! Combining results...")
    else:
        # Calculate batch size
        batch_size = max(1, len(remaining_urls) // NUM_BATCHES)
        if len(remaining_urls) % NUM_BATCHES > 0:
            batch_size += 1
        
        print(f"Processing {len(remaining_urls)} URLs in batches of ~{batch_size}")
        print(f"Estimated total batches: {(len(remaining_urls) + batch_size - 1) // batch_size}")
        
        # Process URLs in batches
        batch_num = len([f for f in os.listdir(OUTPUT_DIR) if f.startswith('batch_')]) + 1 if os.path.exists(OUTPUT_DIR) else 1
        
        for i in range(0, len(remaining_urls), batch_size):
            batch_urls = remaining_urls[i:i + batch_size]
            print(f"\n--- Processing Batch {batch_num} ({len(batch_urls)} URLs) ---")
            
            batch_results = []
            successful = 0
            failed = 0
            
            for j, url in enumerate(batch_urls, 1):
                print(f"  [{j}/{len(batch_urls)}] Processing: {url[:80]}...")
                
                features = process_single_url(url)
                if features is not None:
                    batch_results.append(features)
                    successful += 1
                    
                    # Show feature summary
                    feature_sum = sum(features[color] for color in COLOR_NAMES)
                    dominant_color = max(COLOR_NAMES, key=lambda c: features[c])
                    print(f"    ✓ Success - Sum: {feature_sum:.3f}, Dominant: {dominant_color} ({features[dominant_color]:.3f})")
                else:
                    failed += 1
                    print(f"    ✗ Failed")
                
                # Brief pause to be respectful to servers
                time.sleep(0.1)
            
            # Save batch results
            if batch_results:
                save_batch_results(batch_results, batch_num, OUTPUT_DIR)
                print(f"  Batch {batch_num} summary: {successful} successful, {failed} failed")
            
            batch_num += 1
            
            # Brief pause between batches
            time.sleep(1)
    
    # Combine all results
    print(f"\n=== Combining All Results ===")
    all_results, _ = load_existing_results(OUTPUT_DIR)
    
    if all_results:
        final_df = pd.DataFrame(all_results)
        final_df = final_df.drop_duplicates(subset=['url'])  # Remove any duplicates
        
        # Reorder columns: url first, then colors in consistent order
        column_order = ['url'] + COLOR_NAMES
        final_df = final_df[column_order]
        
        # Save final results
        final_df.to_csv(FINAL_OUTPUT, index=False)
        
        print(f"✅ Final results saved to {FINAL_OUTPUT}")
        print(f"   Total records: {len(final_df)}")
        print(f"   Success rate: {len(final_df)}/{len(all_urls)} ({len(final_df)/len(all_urls)*100:.1f}%)")
        
        # Show feature statistics
        print(f"\n=== Feature Statistics ===")
        for color in COLOR_NAMES:
            values = final_df[color].values
            print(f"  {color:10s}: min={np.min(values):.4f}, max={np.max(values):.4f}, mean={np.mean(values):.4f}, std={np.std(values):.4f}")
        
        # Verify all features are in valid range
        print(f"\n=== Validation ===")
        valid_ranges = True
        for color in COLOR_NAMES:
            values = final_df[color].values
            if np.any(values <= 0) or np.any(values >= 1):
                print(f"  ⚠️  {color}: values outside (0,1) range!")
                valid_ranges = False
            else:
                print(f"  ✅ {color}: all values in (0,1) range")
        
        if valid_ranges:
            print(f"\n🎉 All features are properly normalized between 0 and 1!")
        
    else:
        print("❌ No results to combine!")

if __name__ == "__main__":
    main()