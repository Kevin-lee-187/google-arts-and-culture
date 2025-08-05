# Google Arts & Culture Image URL Scraper

A polite, educational-purpose scraper for extracting stable preview image URLs from Google Arts & Culture artwork pages.

## Overview

This scraper takes a CSV file containing GAC artwork URLs and extracts the best available preview image URL for each artwork, perfect for creating educational materials, lesson slides, or artwork catalogs.

## Features

- **Multiple extraction methods**: Tries Open Graph, Twitter Card, JSON-LD, and image tag fallbacks
- **Polite scraping**: Configurable delays, proper browser headers, and respectful throttling
- **Referer handling**: Properly handles GAC's requirement for Referer headers to avoid 403 errors
- **Robust error handling**: Continues processing even when individual pages fail
- **Optional image downloading**: Can download images directly with proper headers
- **Detailed logging**: Tracks exactly how each image URL was found and verified

## Quick Start

### Option 1: Use the Simple Runner Script

```bash
cd google-arts-and-culture
python run_gac_scraper.py
```

This will guide you through the process interactively.

### Option 2: Direct Command Line Usage

```bash
cd google-arts-and-culture
python src/scraping/gac_image_scraper.py "painting recommendation/GAC_scrape/google_artcsv.csv" --out results.csv
```

### Option 3: With Image Downloads

```bash
python src/scraping/gac_image_scraper.py "painting recommendation/GAC_scrape/google_artcsv.csv" \
  --out results.csv \
  --download-dir downloaded_images \
  --delay 5
```

## Input Format

Your CSV file must have these columns:
- `filename`: The desired filename for the artwork (e.g., "2936.jpg")
- `page`: The GAC artwork URL (e.g., "https://artsandculture.google.com/asset/the-lovers-marc-chagall/jQEveVgIzd6-Og")

Example:
```csv
filename,page
2936.jpg,https://artsandculture.google.com/asset/the-lovers-marc-chagall/jQEveVgIzd6-Og
10356.jpg,https://artsandculture.google.com/asset/a-spiritualistic-s%C3%A9ance-kunnas-v%C3%A4in%C3%B6/BAG6lhZPN6IXzQ
```

## Output Format

The output CSV contains:
- `filename`: Original filename from input
- `page`: GAC artwork page URL
- `image_url`: Extracted preview image URL
- `source`: Method used to find the image (`og:image`, `twitter:image`, `jsonld`, `fallback:img`)
- `status`: `ok` or `error`
- `notes`: Additional details (e.g., "403 on verify (needs Referer)")
- `download`: (if downloading) `ok` or `error`
- `download_notes`: (if downloading) Download status details

## How It Works

### 1. Extraction Strategy

The scraper tries multiple methods in order of reliability:

1. **Open Graph meta tag**: `<meta property="og:image" content="...">`
   - Most reliable for GAC, gives clean preview URLs
2. **Twitter Card meta tag**: `<meta name="twitter:image" content="...">`
   - Good backup method
3. **JSON-LD structured data**: Searches `<script type="application/ld+json">` for image fields
   - Handles complex structured data
4. **Image tag fallback**: Finds the largest `<img>` tag on the page
   - Last resort for pages without proper meta tags

### 2. URL Verification

Each extracted URL is verified with a HEAD/GET request using the original GAC page as the `Referer` header. This is crucial because:

- GAC images (especially on `lh3.googleusercontent.com`) often return 403 without proper Referer
- The scraper tests accessibility and notes when Referer is required
- You can still use URLs marked as "needs Referer" - just set the header when downloading

### 3. Polite Scraping

- **Realistic headers**: Uses proper browser User-Agent and headers
- **Configurable delays**: Default 3-second delay between requests (adjustable)
- **Timeout handling**: 30-second timeout for slow responses
- **Error resilience**: Continues processing even when individual pages fail
- **No retry loops**: Fails gracefully without hammering servers

## Command Line Options

```
python src/scraping/gac_image_scraper.py INPUT_CSV --out OUTPUT_CSV [options]

Required:
  INPUT_CSV              Input CSV file with filename,page columns
  --out OUTPUT_CSV       Output CSV file path

Optional:
  --download-dir DIR     Directory to download images (creates if needed)
  --delay SECONDS        Delay between requests (default: 3.0)
  --timeout SECONDS      Request timeout (default: 30.0)
```

## Examples

### Basic URL extraction:
```bash
python src/scraping/gac_image_scraper.py data.csv --out results.csv
```

### With downloads and custom delay:
```bash
python src/scraping/gac_image_scraper.py data.csv \
  --out results.csv \
  --download-dir images \
  --delay 5
```

### Processing large files with longer timeout:
```bash
python src/scraping/gac_image_scraper.py large_data.csv \
  --out results.csv \
  --timeout 60 \
  --delay 4
```

## Understanding the Results

### Status Codes

- `ok`: Image URL found and verified as accessible
- `error`: Problem occurred (see notes for details)

### Common Notes

- Empty: Image URL works perfectly
- `403 on verify (needs Referer)`: Image exists but requires Referer header
- `page_fetch_failed: HTTP 404`: The GAC page doesn't exist
- `no_image_found`: No suitable image found on the page
- `HTTP 403`: Image server rejected the request

### Source Methods

- `og:image`: Found via Open Graph meta tag (most reliable)
- `twitter:image`: Found via Twitter Card meta tag
- `jsonld`: Found in JSON-LD structured data
- `fallback:img`: Found by analyzing img tags (less reliable)

## Troubleshooting

### "403 on verify (needs Referer)" Images

These images are still usable - just set the Referer header when downloading:

```python
import requests

headers = {'Referer': 'https://artsandculture.google.com/asset/artwork-url'}
response = requests.get(image_url, headers=headers)
```

### Rate Limiting

If you get frequent timeouts or 429 errors:
- Increase the delay: `--delay 10`
- Run during off-peak hours
- Process smaller batches

### Missing Images

Some artworks might not have suitable preview images:
- Check if the GAC page loads properly in a browser
- Some pages might only have thumbnail images
- Private or restricted artworks might not expose images

## Best Practices

1. **Start small**: Test with a few URLs before processing thousands
2. **Use appropriate delays**: 3-5 seconds is usually good for GAC
3. **Check your results**: Review the output CSV for errors and success rates
4. **Respect robots.txt**: This scraper is designed for educational fair use
5. **Save intermediate results**: For large jobs, consider processing in batches

## Educational Use

This scraper is designed specifically for educational purposes:
- Creating lesson materials and presentations
- Building art history databases for teaching
- Research and academic projects
- Fair use educational content

Always respect copyright and terms of service when using the extracted images.

## Dependencies

The scraper requires:
- `requests`: HTTP requests
- `beautifulsoup4`: HTML parsing
- `pathlib`: File path handling (built-in)
- `csv`, `json`, `re`, `time`, `urllib.parse`: Standard library

These are already included in the project's requirements.txt.

## License

This scraper is provided for educational purposes. Respect the terms of service of Google Arts & Culture and copyright holders of the artworks.