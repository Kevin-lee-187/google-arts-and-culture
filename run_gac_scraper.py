#!/usr/bin/env python3
"""
Simple runner script for the GAC image scraper.
Designed to work with the existing google_artcsv.csv file.
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import our scraper
sys.path.append(str(Path(__file__).parent / "src"))

from scraping.gac_image_scraper import GACImageScraper


def main():
    """Run the scraper on the existing CSV file."""
    
    # File paths
    input_csv = Path("painting recommendation/GAC_scrape/google_artcsv.csv")
    output_csv = Path("painting recommendation/GAC_scrape/google_artcsv_with_images.csv")
    download_dir = Path("painting recommendation/GAC_scrape/downloaded_images")
    
    # Check if input file exists
    if not input_csv.exists():
        print(f"Error: Input file not found: {input_csv}")
        print("Make sure you're running this from the google-arts-and-culture directory")
        return 1
    
    print("GAC Image URL Scraper")
    print("====================")
    print(f"Input:  {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Downloads: {download_dir} (optional)")
    print()
    
    # Ask user for preferences
    print("Options:")
    print("1. Extract URLs only (fast)")
    print("2. Extract URLs and download images (slow)")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    download_images = choice == "2"
    if download_images:
        download_dir.mkdir(parents=True, exist_ok=True)
        print(f"Images will be downloaded to: {download_dir}")
    else:
        download_dir = None
        print("Images will NOT be downloaded")
    
    # Get delay setting
    delay_input = input("Delay between requests in seconds (default 3.0): ").strip()
    try:
        delay = float(delay_input) if delay_input else 3.0
    except ValueError:
        delay = 3.0
    
    print(f"Using {delay}s delay between requests")
    print()
    
    # Create and run scraper
    try:
        scraper = GACImageScraper(delay=delay)
        scraper.process_csv(input_csv, output_csv, download_dir)
        
        print("\n" + "="*50)
        print("SUCCESS!")
        print(f"Results saved to: {output_csv}")
        if download_images:
            print(f"Images downloaded to: {download_dir}")
        print("\nThe output CSV contains:")
        print("- filename: original filename from input")
        print("- page: GAC artwork page URL")
        print("- image_url: extracted preview image URL")
        print("- source: method used to find the image (og:image, twitter:image, etc.)")
        print("- status: ok or error")
        print("- notes: additional details (e.g., '403 on verify (needs Referer)')")
        if download_images:
            print("- download: ok or error")
            print("- download_notes: download status details")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())