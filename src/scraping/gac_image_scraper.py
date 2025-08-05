#!/usr/bin/env python3
"""
Google Arts & Culture Image URL Scraper

A polite scraper that extracts stable preview image URLs from GAC artwork pages.
Designed for educational purposes with proper throttling and header handling.

Usage:
    python gac_image_scraper.py input.csv --out output.csv [options]
"""

import argparse
import csv
import json
import re
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup, Tag


class GACImageScraper:
    """Scraper for Google Arts & Culture image URLs."""
    
    def __init__(self, delay: float = 3.0, timeout: float = 30.0):
        """Initialize the scraper with configurable delays and timeouts."""
        self.delay = delay
        self.timeout = timeout
        self.session = self._create_session()
        
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
    
    def fetch_page(self, url: str) -> Tuple[Optional[str], int, str]:
        """
        Fetch HTML content from a URL.
        
        Returns:
            Tuple of (html_content, status_code, error_message)
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code == 200:
                return response.text, response.status_code, ""
            else:
                return None, response.status_code, f"HTTP {response.status_code}"
        except requests.exceptions.RequestException as e:
            return None, 0, str(e)
    
    def extract_image_url(self, html: str, page_url: str) -> Tuple[Optional[str], str]:
        """
        Extract the best candidate image URL from HTML.
        
        Tries in order:
        1. Open Graph meta tag
        2. Twitter Card meta tag  
        3. JSON-LD structured data
        4. Fallback to largest img tag
        
        Returns:
            Tuple of (image_url, source_method)
        """
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
            # Check for direct image field
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
            
            # Recursively search nested objects
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
            
            # Skip tiny images (likely icons or UI elements)
            width = self._parse_dimension(img.get('width'))
            height = self._parse_dimension(img.get('height'))
            
            if width and height:
                size = width * height
                if size > best_size and size > 10000:  # At least 100x100
                    best_size = size
                    best_img = img
            elif not best_img:  # Fallback to first reasonable image
                best_img = img
        
        return best_img
    
    def _parse_dimension(self, dim_str: Optional[str]) -> Optional[int]:
        """Parse dimension string to integer."""
        if not dim_str:
            return None
        try:
            # Handle "500px" or "500"
            return int(re.sub(r'[^\d]', '', str(dim_str)))
        except (ValueError, TypeError):
            return None
    
    def _normalize_url(self, url: str, base_url: str) -> str:
        """Convert relative URLs to absolute URLs."""
        if not url:
            return url
        return urllib.parse.urljoin(base_url, url)
    
    def verify_image_url(self, image_url: str, referer: str) -> Tuple[str, str]:
        """
        Verify that the image URL is accessible.
        
        Returns:
            Tuple of (status, notes)
        """
        if not image_url:
            return 'error', 'no_url'
        
        try:
            # Make a HEAD request with proper referer
            headers = {'Referer': referer}
            response = self.session.head(image_url, headers=headers, timeout=self.timeout)
            
            if response.status_code == 200:
                return 'ok', ''
            elif response.status_code == 403:
                return 'ok', '403 on verify (needs Referer)'
            elif response.status_code == 405:
                # Some servers don't support HEAD, try GET
                response = self.session.get(image_url, headers=headers, timeout=self.timeout, stream=True)
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
    
    def download_image(self, image_url: str, referer: str, filepath: Path) -> Tuple[str, str]:
        """
        Download image to specified filepath.
        
        Returns:
            Tuple of (download_status, download_notes)
        """
        if not image_url:
            return 'error', 'no_url'
        
        try:
            headers = {'Referer': referer}
            response = self.session.get(image_url, headers=headers, timeout=self.timeout)
            
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
    
    def process_csv(self, input_file: Path, output_file: Path, 
                   download_dir: Optional[Path] = None) -> None:
        """
        Process CSV file and extract image URLs.
        
        Args:
            input_file: Input CSV with filename,page columns
            output_file: Output CSV path
            download_dir: Optional directory to download images
        """
        # Validate input file
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Read and validate CSV structure
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if 'filename' not in reader.fieldnames or 'page' not in reader.fieldnames:
                raise ValueError("Input CSV must have 'filename' and 'page' columns")
            
            rows = list(reader)
        
        print(f"Processing {len(rows)} rows from {input_file}")
        
        # Prepare output file
        fieldnames = ['filename', 'page', 'image_url', 'source', 'status', 'notes']
        if download_dir:
            fieldnames.extend(['download', 'download_notes'])
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, row in enumerate(rows, 1):
                print(f"\nProcessing {i}/{len(rows)}: {row['filename']}")
                
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
                    writer.writerow(result)
                    continue
                
                # Fetch page HTML
                html, status_code, error_msg = self.fetch_page(row['page'])
                
                if html is None:
                    result.update({
                        'status': 'error',
                        'notes': f'page_fetch_failed: {error_msg}'
                    })
                    writer.writerow(result)
                    time.sleep(self.delay)
                    continue
                
                # Extract image URL
                image_url, source = self.extract_image_url(html, row['page'])
                
                if image_url is None:
                    result.update({
                        'status': 'error',
                        'source': source,
                        'notes': 'no_image_found'
                    })
                    writer.writerow(result)
                    time.sleep(self.delay)
                    continue
                
                # Verify image URL
                verify_status, verify_notes = self.verify_image_url(image_url, row['page'])
                
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
                        image_url, row['page'], download_path
                    )
                    result.update({
                        'download': download_status,
                        'download_notes': download_notes
                    })
                
                writer.writerow(result)
                
                # Polite delay between requests
                if i < len(rows):  # Don't delay after the last request
                    print(f"Waiting {self.delay}s before next request...")
                    time.sleep(self.delay)
        
        print(f"\nCompleted! Results saved to {output_file}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Extract image URLs from Google Arts & Culture pages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gac_image_scraper.py input.csv --out results.csv
  python gac_image_scraper.py data.csv --out results.csv --download-dir images/ --delay 5
        """
    )
    
    parser.add_argument('input_csv', type=Path,
                       help='Input CSV file with filename,page columns')
    parser.add_argument('--out', type=Path, required=True,
                       help='Output CSV file path')
    parser.add_argument('--download-dir', type=Path,
                       help='Directory to download images (optional)')
    parser.add_argument('--delay', type=float, default=3.0,
                       help='Delay between requests in seconds (default: 3.0)')
    parser.add_argument('--timeout', type=float, default=30.0,
                       help='Request timeout in seconds (default: 30.0)')
    
    args = parser.parse_args()
    
    # Create scraper and process
    scraper = GACImageScraper(delay=args.delay, timeout=args.timeout)
    
    try:
        scraper.process_csv(args.input_csv, args.out, args.download_dir)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())