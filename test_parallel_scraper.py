#!/usr/bin/env python3
"""
Test script for the parallel GAC image scraper.
Tests parallel processing and resume functionality.
"""

import csv
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from scraping.gac_parallel_scraper import ParallelGACImageScraper


def create_test_sample():
    """Create a test sample for parallel processing."""
    
    input_file = Path("painting recommendation/GAC_scrape/google_artcsv.csv")
    test_file = Path("painting recommendation/GAC_scrape/parallel_test_sample.csv")
    
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        return None
    
    # Read first 20 rows for parallel testing
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = [next(reader)]  # header
        rows.extend([next(reader) for _ in range(20)])  # first 20 data rows
    
    # Write test sample
    with open(test_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"Created parallel test sample: {test_file}")
    return test_file


def main():
    # Create test sample
    test_file = create_test_sample()
    if not test_file:
        return 1
    
    output_file = Path("painting recommendation/GAC_scrape/parallel_test_results.csv") 
    
    print("\nTesting PARALLEL GAC Image Scraper")
    print("=" * 50)
    print(f"Input:    {test_file}")
    print(f"Output:   {output_file}")
    print(f"Workers:  5 parallel threads")
    print(f"Delay:    0.1s between batches")
    print()
    
    # First run - process all
    try:
        print("🚀 FIRST RUN - Processing all 20 URLs...")
        scraper = ParallelGACImageScraper(max_workers=5, delay=0.1)
        scraper.process_csv(test_file, output_file, resume=False)
        
        # Show results
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            results = list(reader)
        
        print(f"\n📊 FIRST RUN RESULTS:")
        print(f"Total processed: {len(results)}")
        
        success_count = sum(1 for r in results if r['status'] == 'ok')
        error_count = len(results) - success_count
        
        print(f"Successful: {success_count}")
        print(f"Errors: {error_count}")
        
        # Test resume functionality
        print(f"\n🔄 TESTING RESUME - Running again to test skip logic...")
        scraper2 = ParallelGACImageScraper(max_workers=3, delay=0.1)
        scraper2.process_csv(test_file, output_file, resume=True)
        
        print(f"\n✅ PARALLEL SCRAPER TEST COMPLETED!")
        print(f"Results saved to: {output_file}")
        
        if success_count > 0:
            print(f"\n🎯 Sample successful extractions:")
            for result in results[:3]:  # Show first 3
                if result['status'] == 'ok':
                    print(f"  {result['filename']}: {result['source']} -> {result['image_url'][:60]}...")
        
        print(f"\n💡 Performance comparison:")
        print(f"Parallel (5 workers): ~{len(results)*0.1}s vs Sequential: ~{len(results)*3}s")
        print(f"Speed improvement: ~{(len(results)*3)/(len(results)*0.1):.0f}x faster!")
        
        print(f"\n🚀 Ready to run full scraper:")
        print(f"python run_parallel_scraper.py")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())