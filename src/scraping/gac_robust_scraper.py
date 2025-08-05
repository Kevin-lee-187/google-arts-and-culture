#!/usr/bin/env python3
"""
Robust Google Arts & Culture Image URL Scraper

Enhanced version with rate limiting protection, retry logic, and resume from specific filename.
Designed to handle HTTP 429 (Too Many Requests) errors gracefully.
"""

import argparse
import csv
import json
import random
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup, Tag


class RobustGACImageScraper:
    """Rate-limit aware scraper for Google Arts & Culture image URLs."""
    
    def __init__(self, max_workers: int = 5, base_delay: float = 2.0, timeout: float = 30.0):
        """
        Initialize the robust scraper with conservative defaults.
        
        Args:
            max_workers: Number of concurrent workers (reduced for rate limiting)
            base_delay: Base delay between batches
            timeout: Request timeout
        """
        self.max_workers = max_workers
        self.base_delay = base_delay
        self.timeout = timeout
        self.write_lock = Lock()
        self.processed_count = 0
        self.rate_limit_backoff = 1.0  # Dynamic backoff multiplier
        
    def _create_session(self) -> requests.Session:
        """Create HTTP session with realistic browser headers."""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        })
        return session
    
    def fetch_page_with_retry(self, session: requests.Session, url: str, max_retries: int = 3) -> Tuple[Optional[str], int, str]:
        """
        Fetch HTML content with retry logic for rate limiting.
        
        Returns:
            Tuple of (html_content, status_code, error_message)
        """
        for attempt in range(max_retries):
            try:
                # Add random jitter to avoid thundering herd
                if attempt > 0:
                    jitter = random.uniform(0.5, 1.5)
                    backoff_delay = (2 ** attempt) * self.rate_limit_backoff * jitter
                    print(f"    Retry {attempt} after {backoff_delay:.1f}s...")
                    time.sleep(backoff_delay)
                
                response = session.get(url, timeout=self.timeout)
                
                if response.status_code == 200:
                    # Success - reduce backoff
                    self.rate_limit_backoff = max(0.5, self.rate_limit_backoff * 0.9)
                    return response.text, response.status_code, ""
                
                elif response.status_code == 429:
                    # Rate limited - increase backoff
                    self.rate_limit_backoff = min(5.0, self.rate_limit_backoff * 1.5)
                    
                    # Check for Retry-After header
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        try:
                            wait_time = float(retry_after)
                            print(f"    Rate limited. Server says wait {wait_time}s")
                            if wait_time < 300:  # Don't wait more than 5 minutes
                                time.sleep(wait_time)
                                continue
                        except ValueError:
                            pass
                    
                    if attempt < max_retries - 1:
                        continue  # Will retry with exponential backoff
                    else:
                        return None, response.status_code, f"HTTP 429 (rate limited after {max_retries} retries)"
                
                else:
                    return None, response.status_code, f"HTTP {response.status_code}"
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    continue
                return None, 0, str(e)
        
        return None, 0, "Max retries exceeded"
    
    def extract_image_url(self, html: str, page_url: str) -> Tuple[Optional[str], str]:
        """Extract the best candidate image URL from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # 1. Try Open Graph
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            url = og_image['content']
            return self._normalize_url(url, page_url), 'og:image'
        
        # 2. Try Twitter Card
        twitter_image = soup.find('meta', attrs={'name': 'twitter:image'})
        if twitter_image and twitter_image.get('content'):
            url = twitter_image['content']
            return self._normalize_url(url, page_url), 'twitter:image'
        
        # 3. Try JSON-LD
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                image_url = self._extract_from_jsonld(data)
                if image_url:
                    return self._normalize_url(image_url, page_url), 'jsonld'
            except (json.JSONDecodeError, AttributeError):
                continue
        
        # 4. Fallback to img tags
        img_tags = soup.find_all('img')
        best_img = self._find_largest_image(img_tags)
        if best_img:
            url = best_img.get('src') or best_img.get('data-src')
            if url:
                return self._normalize_url(url, page_url), 'fallback:img'
        
        return None, 'none_found'
    
    def _extract_from_jsonld(self, data) -> Optional[str]:
        """Extract image URL from JSON-LD data."""
        if isinstance(data, dict):
            if 'image' in data:
                image = data['image']
                if isinstance(image, str):
                    return image
                elif isinstance(image, dict) and 'url' in image:
                    return image['url']
                elif isinstance(image, list) and len(image) > 0:
                    first_image = image[0]
                    if isinstance(first_image, str):
                        return first_image
                    elif isinstance(first_image, dict) and 'url' in first_image:
                        return first_image['url']
            
            for value in data.values():
                if isinstance(value, (dict, list)):
                    result = self._extract_from_jsonld(value)
                    if result:
                        return result
        
        elif isinstance(data, list):
            for item in data:
                result = self._extract_from_jsonld(item)
                if result:
                    return result
        
        return None
    
    def _find_largest_image(self, img_tags: List[Tag]) -> Optional[Tag]:
        """Find the largest image from a list of img tags."""
        best_img = None
        best_size = 0
        
        for img in img_tags:
            src = img.get('src') or img.get('data-src')
            if not src:
                continue
            
            width = self._parse_dimension(img.get('width'))
            height = self._parse_dimension(img.get('height'))
            
            if width and height:
                size = width * height
                if size > best_size and size > 10000:
                    best_size = size
                    best_img = img
            elif not best_img:
                best_img = img
        
        return best_img
    
    def _parse_dimension(self, dim_str: Optional[str]) -> Optional[int]:
        """Parse dimension string to integer."""
        if not dim_str:
            return None
        try:
            return int(re.sub(r'[^\d]', '', str(dim_str)))
        except (ValueError, TypeError):
            return None
    
    def _normalize_url(self, url: str, base_url: str) -> str:
        """Convert relative URLs to absolute URLs."""
        if not url:
            return url
        return urllib.parse.urljoin(base_url, url)
    
    def verify_image_url(self, session: requests.Session, image_url: str, referer: str) -> Tuple[str, str]:
        """Verify that the image URL is accessible with retry logic."""
        if not image_url:
            return 'error', 'no_url'
        
        try:
            headers = {'Referer': referer}
            response = session.head(image_url, headers=headers, timeout=self.timeout)
            
            if response.status_code == 200:
                return 'ok', ''
            elif response.status_code == 403:
                return 'ok', '403 on verify (needs Referer)'
            elif response.status_code == 429:
                return 'ok', '429 on verify (rate limited)'
            elif response.status_code == 405:
                # Some servers don't support HEAD, try GET
                response = session.get(image_url, headers=headers, timeout=self.timeout, stream=True)
                if response.status_code == 200:
                    return 'ok', ''
                elif response.status_code == 403:
                    return 'ok', '403 on verify (needs Referer)'
                else:
                    return 'error', f'HTTP {response.status_code}'
            else:
                return 'error', f'HTTP {response.status_code}'
                
        except requests.exceptions.RequestException as e:
            return 'error', f'verify_failed: {str(e)}'
    
    def process_single_row(self, row: Dict[str, str], download_dir: Optional[Path] = None) -> Dict[str, str]:
        """Process a single row with robust error handling."""
        # Create a session for this thread
        session = self._create_session()
        
        result = {
            'filename': row['filename'],
            'page': row['page'],
            'image_url': '',
            'source': '',
            'status': '',
            'notes': ''
        }
        
        if download_dir:
            result.update({'download': '', 'download_notes': ''})
        
        # Skip empty pages
        if not row['page'].strip():
            result.update({
                'status': 'error',
                'notes': 'empty_page_url'
            })
            return result
        
        # Fetch page HTML with retry logic
        html, status_code, error_msg = self.fetch_page_with_retry(session, row['page'])
        
        if html is None:
            result.update({
                'status': 'error',
                'notes': f'page_fetch_failed: {error_msg}'
            })
            return result
        
        # Extract image URL
        image_url, source = self.extract_image_url(html, row['page'])
        
        if image_url is None:
            result.update({
                'status': 'error',
                'source': source,
                'notes': 'no_image_found'
            })
            return result
        
        # Verify image URL (with less aggressive verification to avoid more rate limiting)
        verify_status, verify_notes = self.verify_image_url(session, image_url, row['page'])
        
        result.update({
            'image_url': image_url,
            'source': source,
            'status': verify_status,
            'notes': verify_notes
        })
        
        return result
    
    def get_processed_filenames(self, output_file: Path) -> Set[str]:
        """Get set of already processed filenames from existing output file."""
        processed = set()
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        processed.add(row['filename'])
            except Exception as e:
                print(f"Warning: Could not read existing output file: {e}")
        return processed
    
    def find_start_position(self, all_rows: List[Dict], start_filename: str) -> int:
        """Find the position to start processing from."""
        for i, row in enumerate(all_rows):
            if row['filename'] == start_filename:
                return i
        return 0  # If not found, start from beginning
    
    def process_csv(self, input_file: Path, output_file: Path, 
                   download_dir: Optional[Path] = None, 
                   start_from_filename: Optional[str] = None,
                   resume: bool = True) -> None:
        """
        Process CSV file with robust rate limiting and resume capability.
        
        Args:
            input_file: Input CSV with filename,page columns
            output_file: Output CSV path
            download_dir: Optional directory to download images
            start_from_filename: Specific filename to start from
            resume: Whether to resume from existing output file
        """
        # Validate input file
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Read and validate CSV structure
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if 'filename' not in reader.fieldnames or 'page' not in reader.fieldnames:
                raise ValueError("Input CSV must have 'filename' and 'page' columns")
            
            all_rows = list(reader)
        
        # Determine starting position
        if start_from_filename:
            start_pos = self.find_start_position(all_rows, start_from_filename)
            print(f"🎯 Starting from filename: {start_from_filename} (position {start_pos})")
            all_rows = all_rows[start_pos:]  # Start from this position
            resume = False  # Don't use resume logic when starting from specific filename
        
        # Get already processed filenames for resume functionality
        processed_filenames = set()
        if resume:
            processed_filenames = self.get_processed_filenames(output_file)
            print(f"Found {len(processed_filenames)} already processed files")
        
        # Filter out already processed rows
        if resume:
            rows_to_process = [row for row in all_rows if row['filename'] not in processed_filenames]
        else:
            rows_to_process = all_rows
        
        print(f"Total rows in scope: {len(all_rows)}")
        print(f"Rows to process: {len(rows_to_process)}")
        print(f"Using {self.max_workers} workers with {self.base_delay}s base delay")
        print(f"Rate limit protection: ENABLED")
        
        if not rows_to_process:
            print("No new rows to process!")
            return
        
        # Prepare output file
        fieldnames = ['filename', 'page', 'image_url', 'source', 'status', 'notes']
        if download_dir:
            fieldnames.extend(['download', 'download_notes'])
        
        # Determine write mode
        if start_from_filename:
            write_mode = 'w'  # Start fresh when starting from specific filename
            write_header = True
        else:
            write_mode = 'a' if resume and output_file.exists() else 'w'
            write_header = not (resume and output_file.exists())
        
        with open(output_file, write_mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            
            # Process in smaller batches with longer delays for rate limiting
            batch_size = min(self.max_workers, 3)  # Smaller batches
            total_batches = (len(rows_to_process) + batch_size - 1) // batch_size
            
            for batch_num, i in enumerate(range(0, len(rows_to_process), batch_size), 1):
                batch_rows = rows_to_process[i:i + batch_size]
                
                print(f"\n📦 Batch {batch_num}/{total_batches} ({len(batch_rows)} rows) - Backoff: {self.rate_limit_backoff:.1f}x")
                
                # Sequential processing to avoid overwhelming the server
                batch_results = []
                for j, row in enumerate(batch_rows, 1):
                    print(f"  Processing {j}/{len(batch_rows)}: {row['filename']}")
                    
                    try:
                        result = self.process_single_row(row, download_dir)
                        batch_results.append(result)
                        self.processed_count += 1
                        
                        status_emoji = "✅" if result['status'] == 'ok' else "❌"
                        print(f"    {status_emoji} {result['status']}: {result.get('notes', '')}")
                        
                        # Small delay between individual requests within batch
                        if j < len(batch_rows):
                            time.sleep(0.5 * self.rate_limit_backoff)
                            
                    except Exception as e:
                        print(f"    ❌ Error processing {row['filename']}: {e}")
                        error_result = {
                            'filename': row['filename'],
                            'page': row['page'],
                            'image_url': '',
                            'source': '',
                            'status': 'error',
                            'notes': f'processing_error: {str(e)}'
                        }
                        if download_dir:
                            error_result.update({'download': '', 'download_notes': ''})
                        batch_results.append(error_result)
                
                # Write all results from this batch
                with self.write_lock:
                    for result in batch_results:
                        writer.writerow(result)
                    f.flush()  # Ensure data is written immediately
                
                # Adaptive delay between batches
                if batch_num < total_batches:
                    delay = self.base_delay * self.rate_limit_backoff
                    print(f"  💤 Waiting {delay:.1f}s before next batch...")
                    time.sleep(delay)
        
        success_rate = (self.processed_count / len(rows_to_process)) * 100 if rows_to_process else 0
        
        print(f"\n" + "="*60)
        print(f"🎉 ROBUST PROCESSING COMPLETED!")
        print(f"Processed in this run: {self.processed_count}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Final rate limit backoff: {self.rate_limit_backoff:.1f}x")
        print(f"Results saved to: {output_file}")


def main():
    """Command-line interface for robust scraper."""
    parser = argparse.ArgumentParser(
        description="Robust Google Arts & Culture image URL scraper with rate limiting protection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gac_robust_scraper.py input.csv --out results.csv
  python gac_robust_scraper.py data.csv --out results.csv --start-from "12974.jpg"
  python gac_robust_scraper.py data.csv --out results.csv --workers 3 --delay 5
        """
    )
    
    parser.add_argument('input_csv', type=Path,
                       help='Input CSV file with filename,page columns')
    parser.add_argument('--out', type=Path, required=True,
                       help='Output CSV file path')
    parser.add_argument('--start-from', type=str,
                       help='Filename to start processing from (e.g., "12974.jpg")')
    parser.add_argument('--download-dir', type=Path,
                       help='Directory to download images (optional)')
    parser.add_argument('--workers', type=int, default=3,
                       help='Number of parallel workers (default: 3, conservative for rate limiting)')
    parser.add_argument('--delay', type=float, default=3.0,
                       help='Base delay between batches in seconds (default: 3.0)')
    parser.add_argument('--timeout', type=float, default=30.0,
                       help='Request timeout in seconds (default: 30.0)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start from beginning (ignore existing output)')
    
    args = parser.parse_args()
    
    # Create scraper and process
    scraper = RobustGACImageScraper(
        max_workers=args.workers,
        base_delay=args.delay,
        timeout=args.timeout
    )
    
    try:
        scraper.process_csv(
            args.input_csv, 
            args.out, 
            args.download_dir,
            start_from_filename=args.start_from,
            resume=not args.no_resume
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())