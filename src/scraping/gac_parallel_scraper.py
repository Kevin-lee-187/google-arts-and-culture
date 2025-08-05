#!/usr/bin/env python3
"""
Parallel Google Arts & Culture Image URL Scraper

Enhanced version with concurrent processing and resume functionality.
Processes multiple URLs simultaneously while maintaining politeness.
"""

import argparse
import csv
import json
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup, Tag


class ParallelGACImageScraper:
    """Parallel scraper for Google Arts & Culture image URLs."""
    
    def __init__(self, max_workers: int = 10, delay: float = 0.3, timeout: float = 30.0):
        """
        Initialize the parallel scraper.
        
        Args:
            max_workers: Number of concurrent workers
            delay: Delay between batches (not individual requests)
            timeout: Request timeout
        """
        self.max_workers = max_workers
        self.delay = delay  # Delay between batches, not individual requests
        self.timeout = timeout
        self.write_lock = Lock()
        self.processed_count = 0
        
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
        })
        return session
    
    def fetch_page(self, session: requests.Session, url: str) -> Tuple[Optional[str], int, str]:
        """Fetch HTML content from a URL."""
        try:
            response = session.get(url, timeout=self.timeout)
            if response.status_code == 200:
                return response.text, response.status_code, ""
            else:
                return None, response.status_code, f"HTTP {response.status_code}"
        except requests.exceptions.RequestException as e:
            return None, 0, str(e)
    
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
        """Verify that the image URL is accessible."""
        if not image_url:
            return 'error', 'no_url'
        
        try:
            headers = {'Referer': referer}
            response = session.head(image_url, headers=headers, timeout=self.timeout)
            
            if response.status_code == 200:
                return 'ok', ''
            elif response.status_code == 403:
                return 'ok', '403 on verify (needs Referer)'
            elif response.status_code == 405:
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
    
    def download_image(self, session: requests.Session, image_url: str, referer: str, filepath: Path) -> Tuple[str, str]:
        """Download image to specified filepath."""
        if not image_url:
            return 'error', 'no_url'
        
        try:
            headers = {'Referer': referer}
            response = session.get(image_url, headers=headers, timeout=self.timeout)
            
            if response.status_code == 200:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                return 'ok', ''
            else:
                return 'error', f'HTTP {response.status_code}'
                
        except requests.exceptions.RequestException as e:
            return 'error', str(e)
        except IOError as e:
            return 'error', f'file_error: {str(e)}'
    
    def process_single_row(self, row: Dict[str, str], download_dir: Optional[Path] = None) -> Dict[str, str]:
        """Process a single row (designed for parallel execution)."""
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
        
        # Fetch page HTML
        html, status_code, error_msg = self.fetch_page(session, row['page'])
        
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
        
        # Verify image URL
        verify_status, verify_notes = self.verify_image_url(session, image_url, row['page'])
        
        result.update({
            'image_url': image_url,
            'source': source,
            'status': verify_status,
            'notes': verify_notes
        })
        
        # Download if requested
        if download_dir and verify_status == 'ok':
            download_path = download_dir / row['filename']
            download_status, download_notes = self.download_image(
                session, image_url, row['page'], download_path
            )
            result.update({
                'download': download_status,
                'download_notes': download_notes
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
    
    def process_csv(self, input_file: Path, output_file: Path, 
                   download_dir: Optional[Path] = None, resume: bool = True) -> None:
        """
        Process CSV file with parallel execution and resume capability.
        
        Args:
            input_file: Input CSV with filename,page columns
            output_file: Output CSV path
            download_dir: Optional directory to download images
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
        
        # Get already processed filenames for resume functionality
        processed_filenames = set()
        if resume:
            processed_filenames = self.get_processed_filenames(output_file)
            print(f"Found {len(processed_filenames)} already processed files")
        
        # Filter out already processed rows
        rows_to_process = [row for row in all_rows if row['filename'] not in processed_filenames]
        
        print(f"Total rows in input: {len(all_rows)}")
        print(f"Rows to process: {len(rows_to_process)}")
        print(f"Using {self.max_workers} parallel workers")
        print(f"Batch delay: {self.delay}s")
        
        if not rows_to_process:
            print("No new rows to process!")
            return
        
        # Prepare output file
        fieldnames = ['filename', 'page', 'image_url', 'source', 'status', 'notes']
        if download_dir:
            fieldnames.extend(['download', 'download_notes'])
        
        # Determine write mode
        write_mode = 'a' if resume and output_file.exists() else 'w'
        write_header = not (resume and output_file.exists())
        
        with open(output_file, write_mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            
            # Process in batches using ThreadPoolExecutor
            batch_size = self.max_workers
            total_batches = (len(rows_to_process) + batch_size - 1) // batch_size
            
            for batch_num, i in enumerate(range(0, len(rows_to_process), batch_size), 1):
                batch_rows = rows_to_process[i:i + batch_size]
                
                print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch_rows)} rows)")
                
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit all jobs in this batch
                    future_to_row = {
                        executor.submit(self.process_single_row, row, download_dir): row 
                        for row in batch_rows
                    }
                    
                    # Collect results as they complete
                    batch_results = []
                    for future in as_completed(future_to_row):
                        row = future_to_row[future]
                        try:
                            result = future.result()
                            batch_results.append(result)
                            self.processed_count += 1
                            
                            # Show progress
                            total_processed = len(processed_filenames) + self.processed_count
                            print(f"  Completed: {result['filename']} ({result['status']}) - Total: {total_processed}/{len(all_rows)}")
                            
                        except Exception as e:
                            print(f"  Error processing {row['filename']}: {e}")
                            # Create error result
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
                        f.flush()  # Ensure data is written
                
                # Delay between batches (not individual requests)
                if batch_num < total_batches:
                    print(f"  Waiting {self.delay}s before next batch...")
                    time.sleep(self.delay)
        
        total_processed = len(processed_filenames) + self.processed_count
        success_rate = (self.processed_count / len(rows_to_process)) * 100 if rows_to_process else 0
        
        print(f"\n" + "="*60)
        print(f"PARALLEL PROCESSING COMPLETED!")
        print(f"Total processed in this run: {self.processed_count}")
        print(f"Total in output file: {total_processed}/{len(all_rows)}")
        print(f"Success rate this run: {success_rate:.1f}%")
        print(f"Results saved to: {output_file}")
        if download_dir:
            print(f"Images downloaded to: {download_dir}")


def main():
    """Command-line interface for parallel scraper."""
    parser = argparse.ArgumentParser(
        description="Parallel Google Arts & Culture image URL scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gac_parallel_scraper.py input.csv --out results.csv
  python gac_parallel_scraper.py data.csv --out results.csv --workers 20 --no-resume
  python gac_parallel_scraper.py data.csv --out results.csv --download-dir images/ --workers 15
        """
    )
    
    parser.add_argument('input_csv', type=Path,
                       help='Input CSV file with filename,page columns')
    parser.add_argument('--out', type=Path, required=True,
                       help='Output CSV file path')
    parser.add_argument('--download-dir', type=Path,
                       help='Directory to download images (optional)')
    parser.add_argument('--workers', type=int, default=10,
                       help='Number of parallel workers (default: 10)')
    parser.add_argument('--delay', type=float, default=0.3,
                       help='Delay between batches in seconds (default: 0.3)')
    parser.add_argument('--timeout', type=float, default=30.0,
                       help='Request timeout in seconds (default: 30.0)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start from beginning (ignore existing output)')
    
    args = parser.parse_args()
    
    # Create scraper and process
    scraper = ParallelGACImageScraper(
        max_workers=args.workers,
        delay=args.delay,
        timeout=args.timeout
    )
    
    try:
        scraper.process_csv(
            args.input_csv, 
            args.out, 
            args.download_dir,
            resume=not args.no_resume
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())