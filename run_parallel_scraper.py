#!/usr/bin/env python3
"""
Parallel runner script for the GAC image scraper.
Automatically resumes from where the previous run was interrupted.
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import our scraper
sys.path.append(str(Path(__file__).parent / "src"))

from scraping.gac_parallel_scraper import ParallelGACImageScraper


def main():
    """Run the parallel scraper on the existing CSV file."""
    
    # File paths
    input_csv = Path("painting recommendation/GAC_scrape/google_artcsv.csv")
    output_csv = Path("painting recommendation/GAC_scrape/google_artcsv_with_images.csv")
    download_dir = Path("painting recommendation/GAC_scrape/downloaded_images")
    
    # Check if input file exists
    if not input_csv.exists():
        print(f"Error: Input file not found: {input_csv}")
        print("Make sure you're running this from the google-arts-and-culture directory")
        return 1
    
    print("PARALLEL GAC Image URL Scraper")
    print("=" * 40)
    print(f"Input:  {input_csv}")
    print(f"Output: {output_csv}")
    print()
    
    # Check if resuming
    if output_csv.exists():
        print("🔄 RESUME MODE: Found existing output file")
        print("Will skip already processed files and continue from where you left off")
        print()
    
    # Ask user for preferences
    print("Options:")
    print("1. Extract URLs only (fast)")
    print("2. Extract URLs and download images (slower)")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    download_images = choice == "2"
    if download_images:
        download_dir.mkdir(parents=True, exist_ok=True)
        print(f"Images will be downloaded to: {download_dir}")
    else:
        download_dir = None
        print("Images will NOT be downloaded")
    
    # Get parallel workers setting
    workers_input = input("Number of parallel workers (default 10, max recommended 20): ").strip()
    try:
        workers = int(workers_input) if workers_input else 10
        workers = min(max(workers, 1), 50)  # Clamp between 1 and 50
    except ValueError:
        workers = 10
    
    # Get batch delay setting
    delay_input = input("Delay between batches in seconds (default 0.3): ").strip()
    try:
        delay = float(delay_input) if delay_input else 0.3
    except ValueError:
        delay = 0.3
    
    print(f"\nConfiguration:")
    print(f"- Workers: {workers} parallel threads")
    print(f"- Batch delay: {delay}s")
    print(f"- Resume: {'Yes' if output_csv.exists() else 'No'}")
    print()
    
    confirm = input("Start processing? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return 0
    
    # Create and run scraper
    try:
        scraper = ParallelGACImageScraper(
            max_workers=workers,
            delay=delay
        )
        scraper.process_csv(input_csv, output_csv, download_dir, resume=True)
        
        print("\n" + "="*60)
        print("🎉 SUCCESS!")
        print(f"Results saved to: {output_csv}")
        if download_images:
            print(f"Images downloaded to: {download_dir}")
        
        print("\n📊 Output CSV contains:")
        print("- filename: original filename from input")
        print("- page: GAC artwork page URL") 
        print("- image_url: extracted preview image URL")
        print("- source: method used to find the image")
        print("- status: ok or error")
        print("- notes: additional details")
        if download_images:
            print("- download: download status")
            print("- download_notes: download details")
        
        print("\n💡 Tips:")
        print("- If interrupted, just run this script again to resume")
        print("- URLs marked 'needs Referer' are still usable with proper headers")
        print("- Check the notes column for any issues")
        
    except KeyboardInterrupt:
        print("\n\n⏸️  INTERRUPTED!")
        print("Don't worry - progress has been saved.")
        print("Run this script again to resume from where you left off.")
        return 0
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())