#!/usr/bin/env python3
"""
Quick resume script - automatically detects where you left off and continues.
Optimized for maximum speed while being respectful to GAC servers.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from scraping.gac_parallel_scraper import ParallelGACImageScraper


def main():
    """Resume scraping with optimal settings."""
    
    # File paths
    input_csv = Path("painting recommendation/GAC_scrape/google_artcsv.csv")
    output_csv = Path("painting recommendation/GAC_scrape/google_artcsv_with_images.csv")
    
    if not input_csv.exists():
        print(f"❌ Error: Input file not found: {input_csv}")
        return 1
    
    print("🚀 FAST RESUME - GAC Image URL Scraper")
    print("=" * 50)
    
    # Check current progress
    scraper = ParallelGACImageScraper()
    processed_filenames = scraper.get_processed_filenames(output_csv)
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1  # Subtract header
    
    remaining = total_rows - len(processed_filenames)
    progress = (len(processed_filenames) / total_rows) * 100
    
    print(f"📊 Progress: {len(processed_filenames):,}/{total_rows:,} ({progress:.1f}%)")
    print(f"⏳ Remaining: {remaining:,} URLs to process")
    
    if remaining == 0:
        print("✅ All URLs already processed!")
        return 0
    
    # Estimate time
    est_time_min = (remaining * 0.3) / 60  # ~0.3s per URL with 10 workers
    print(f"⏱️  Estimated time: ~{est_time_min:.1f} minutes")
    print()
    
    # Optimal settings for speed + politeness
    workers = 15  # Higher for faster processing
    delay = 0.2   # Shorter delay between batches
    
    print(f"⚡ Using SPEED settings:")
    print(f"   - {workers} parallel workers")
    print(f"   - {delay}s delay between batches")
    print(f"   - Resume from existing progress")
    print()
    
    confirm = input("🚀 Continue with fast processing? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Cancelled. Use run_parallel_scraper.py for custom settings.")
        return 0
    
    print("\n" + "="*50)
    print("STARTING PARALLEL PROCESSING...")
    
    try:
        scraper = ParallelGACImageScraper(
            max_workers=workers,
            delay=delay,
            timeout=30.0
        )
        
        scraper.process_csv(
            input_csv, 
            output_csv, 
            download_dir=None,  # URLs only for speed
            resume=True
        )
        
        print("\n" + "🎉" * 20)
        print("SUCCESS! All URLs extracted!")
        print(f"Results: {output_csv}")
        print("🎉" * 20)
        
    except KeyboardInterrupt:
        print("\n\n⏸️  INTERRUPTED AGAIN!")
        print("Progress saved. Run this script again to resume.")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())