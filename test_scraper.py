#!/usr/bin/env python3
"""
Test script for the GAC image scraper.
Creates a small sample from the main CSV and tests the scraper.
"""

import csv
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from scraping.gac_image_scraper import GACImageScraper


def create_test_sample():
    """Create a small test sample from the main CSV."""
    
    input_file = Path("painting recommendation/GAC_scrape/google_artcsv.csv")
    test_file = Path("painting recommendation/GAC_scrape/test_sample.csv")
    
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        return None
    
    # Read first 5 rows (plus header) for testing
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = [next(reader)]  # header
        rows.extend([next(reader) for _ in range(5)])  # first 5 data rows
    
    # Write test sample
    with open(test_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"Created test sample: {test_file}")
    return test_file


def main():
    # Create test sample
    test_file = create_test_sample()
    if not test_file:
        return 1
    
    # Output files
    output_file = Path("painting recommendation/GAC_scrape/test_results.csv") 
    
    print("\nTesting GAC Image Scraper")
    print("=" * 40)
    print(f"Input:  {test_file}")
    print(f"Output: {output_file}")
    print(f"Delay:  1 second (fast for testing)")
    print()
    
    # Run scraper on test sample
    try:
        scraper = GACImageScraper(delay=1.0)  # Faster delay for testing
        scraper.process_csv(test_file, output_file)
        
        print("\n" + "=" * 40)
        print("Test completed successfully!")
        print(f"Check results in: {output_file}")
        
        # Show a summary of results
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            results = list(reader)
        
        print(f"\nResults Summary:")
        print(f"Total processed: {len(results)}")
        
        success_count = sum(1 for r in results if r['status'] == 'ok')
        error_count = len(results) - success_count
        
        print(f"Successful: {success_count}")
        print(f"Errors: {error_count}")
        
        if success_count > 0:
            print("\nSuccessful extractions:")
            for result in results:
                if result['status'] == 'ok':
                    print(f"  {result['filename']}: {result['source']} -> {result['image_url'][:80]}...")
        
        if error_count > 0:
            print("\nErrors:")
            for result in results:
                if result['status'] == 'error':
                    print(f"  {result['filename']}: {result['notes']}")
        
        print(f"\nIf the test looks good, run the full scraper with:")
        print(f"python run_gac_scraper.py")
        
    except Exception as e:
        print(f"Test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())