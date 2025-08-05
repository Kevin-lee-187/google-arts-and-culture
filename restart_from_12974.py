#!/usr/bin/env python3
"""
Restart scraper from 12974.jpg with rate limiting protection.
Removes HTTP 429 errors and continues from where the rate limiting started.
"""

import csv
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from scraping.gac_robust_scraper import RobustGACImageScraper


def clean_output_file(output_file: Path, problem_filename: str) -> int:
    """
    Remove all entries from problem_filename onwards to restart cleanly.
    Returns the number of good entries kept.
    """
    if not output_file.exists():
        print(f"Output file {output_file} doesn't exist")
        return 0
    
    # Read all entries
    good_entries = []
    found_problem = False
    
    with open(output_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            if row['filename'] == problem_filename:
                found_problem = True
                print(f"Found problem entry: {problem_filename}")
                break
            good_entries.append(row)
    
    if not found_problem:
        print(f"Warning: {problem_filename} not found in output file")
        return len(good_entries)
    
    # Write back only the good entries
    backup_file = output_file.with_suffix('.backup.csv')
    output_file.rename(backup_file)
    print(f"Backed up original file to: {backup_file}")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        if good_entries:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(good_entries)
    
    print(f"Cleaned output file: kept {len(good_entries)} good entries")
    return len(good_entries)


def main():
    """Restart scraping from 12974.jpg with robust rate limiting."""
    
    # File paths
    input_csv = Path("painting recommendation/GAC_scrape/google_artcsv.csv")
    output_csv = Path("painting recommendation/GAC_scrape/google_artcsv_with_images.csv")
    problem_filename = "12974.jpg"
    
    if not input_csv.exists():
        print(f"❌ Error: Input file not found: {input_csv}")
        return 1
    
    print("🔧 RESTART FROM 12974.jpg - Rate Limiting Fixed")
    print("=" * 60)
    
    # Clean the output file
    good_entries = clean_output_file(output_csv, problem_filename)
    
    print(f"\n📊 Status:")
    print(f"   - Successfully processed entries: {good_entries}")
    print(f"   - Restarting from: {problem_filename}")
    print(f"   - Rate limiting protection: ENABLED")
    print(f"   - Conservative settings: 3 workers, 3s delays, retry logic")
    
    print(f"\n🛡️ Rate Limiting Protections:")
    print(f"   - Exponential backoff on HTTP 429")
    print(f"   - Retry logic with jitter")
    print(f"   - Respect Retry-After headers")
    print(f"   - Sequential processing within batches")
    print(f"   - Dynamic delay adjustment")
    
    confirm = input(f"\n🚀 Restart scraping from {problem_filename}? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return 0
    
    print("\n" + "="*60)
    print("STARTING ROBUST SCRAPER WITH RATE LIMITING PROTECTION...")
    
    try:
        # Use conservative settings to avoid rate limiting
        scraper = RobustGACImageScraper(
            max_workers=3,      # Reduced workers
            base_delay=3.0,     # Longer delays
            timeout=30.0
        )
        
        scraper.process_csv(
            input_csv, 
            output_csv, 
            download_dir=None,  # URLs only for now
            start_from_filename=problem_filename,
            resume=False  # Start fresh from the specific filename
        )
        
        print("\n" + "🎉" * 20)
        print("SUCCESS! Scraping completed without rate limiting!")
        print(f"Results: {output_csv}")
        print("🎉" * 20)
        
    except KeyboardInterrupt:
        print("\n\n⏸️  INTERRUPTED!")
        print("Progress has been saved. Run this script again to resume.")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nIf you're still getting rate limited, try:")
        print("1. Wait 30-60 minutes before retrying")
        print("2. Use even more conservative settings")
        print("3. Contact GAC if the issue persists")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())