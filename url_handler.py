import os
import re
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Set
from urllib.parse import urlparse, unquote, urljoin
import requests
from bs4 import BeautifulSoup

from utils import get_sandbox_path


logger = logging.getLogger(__name__)


def extract_urls(text: str) -> Dict[str, List[str]]:
    """
    Extract file URLs and web URLs from text.
    
    Returns:
        dict with keys 'web_urls' and 'file_urls'
    """
    web_urls = []
    file_urls = []
    
    # Pattern for web URLs (http/https)
    web_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    web_matches = re.findall(web_pattern, text)
    web_urls.extend(web_matches)
    
    # Pattern for file:// URLs
    file_url_pattern = r'file://[^\s<>"{}|\\^`\[\]]+'
    file_url_matches = re.findall(file_url_pattern, text)
    
    # Convert file:// URLs to paths
    for url in file_url_matches:
        path = unquote(url.replace('file://', ''))
        file_urls.append(path)
    
    # Pattern for absolute paths (Unix and Windows)
    # Unix: /path/to/file
    # Windows: C:\path\to\file or C:/path/to/file
    unix_path_pattern = r'(?:^|\s)(/[^\s<>"{}|\\^`\[\]]+)'
    windows_path_pattern = r'(?:^|\s)([A-Za-z]:[/\\][^\s<>"{}|\\^`\[\]]*)'
    
    unix_matches = re.findall(unix_path_pattern, text)
    windows_matches = re.findall(windows_path_pattern, text)
    
    # Validate that these are actual file paths
    for path in unix_matches + windows_matches:
        path = path.strip()
        if os.path.exists(path):
            file_urls.append(path)
    
    return {
        'web_urls': list(set(web_urls)),  # Remove duplicates
        'file_urls': list(set(file_urls))
    }


def extract_links_from_html(html_content: str, base_url: str) -> List[str]:
    """
    Extract all links from HTML content.

    Args:
        html_content: HTML content as string
        base_url: Base URL to resolve relative links

    Returns:
        List of absolute URLs found in the HTML
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []

        # Find all <a> tags with href
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            # Only include http/https URLs
            if absolute_url.startswith(('http://', 'https://')):
                links.append(absolute_url)

        return links
    except Exception as e:
        logger.error(f"Error extracting links: {e}")
        return []


def download_web_url(url: str, sandbox_path: str, recursive: bool = False, max_depth: int = 1,
                     _visited: Set[str] = None, _current_depth: int = 0) -> str:
    """
    Download web content to sandbox with optional recursive downloading.

    Args:
        url: URL to download
        sandbox_path: Path to sandbox directory
        recursive: If True, download linked pages (default: False)
        max_depth: Maximum recursion depth for recursive downloads (default: 1)
        _visited: Internal parameter to track visited URLs
        _current_depth: Internal parameter to track current depth

    Returns:
        Path to downloaded file relative to sandbox, or error message
    """
    # Initialize visited set for top-level call
    if _visited is None:
        _visited = set()

    # Skip if already visited or max depth reached
    if url in _visited or _current_depth > max_depth:
        return f"Skipped (already visited or max depth): {url}"

    _visited.add(url)

    try:
        # Parse URL to get filename
        parsed = urlparse(url)
        domain = parsed.netloc
        path_parts = parsed.path.strip('/').split('/')

        # Create subdirectory for domain
        download_dir = os.path.join(sandbox_path, 'downloads', domain)
        os.makedirs(download_dir, exist_ok=True)

        # Determine filename
        if path_parts and path_parts[-1]:
            filename = unquote(path_parts[-1])
            # If no extension, add .html
            if '.' not in filename:
                filename += '.html'
        else:
            # Use hash of URL as filename
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f'page_{url_hash}.html'

        # Handle duplicate filenames by appending depth
        if _current_depth > 0:
            name, ext = os.path.splitext(filename)
            filename = f"{name}_depth{_current_depth}{ext}"

        # Download with timeout and User-Agent header
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=30, allow_redirects=True, headers=headers)
        response.raise_for_status()

        # Save to file
        file_path = os.path.join(download_dir, filename)

        # Handle binary vs text content
        content_type = response.headers.get('content-type', '').lower()
        is_html = 'text' in content_type or 'html' in content_type

        if is_html or 'json' in content_type:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
        else:
            with open(file_path, 'wb') as f:
                f.write(response.content)

        # Return relative path
        rel_path = os.path.relpath(file_path, sandbox_path)
        logger.info(f"Downloaded {url} to {rel_path}")

        # Recursive download if enabled and this is HTML
        if recursive and is_html and _current_depth < max_depth:
            try:
                links = extract_links_from_html(response.text, url)
                # Filter to same domain only
                same_domain_links = [link for link in links if urlparse(link).netloc == domain]

                logger.info(f"Found {len(same_domain_links)} same-domain links at depth {_current_depth}")

                # Download linked pages
                for link in same_domain_links[:50]:  # Limit to 50 links per page to avoid excessive downloads
                    try:
                        download_web_url(link, sandbox_path, recursive=True, max_depth=max_depth,
                                       _visited=_visited, _current_depth=_current_depth + 1)
                    except Exception as e:
                        logger.error(f"Error downloading linked page {link}: {e}")
                        continue
            except Exception as e:
                logger.error(f"Error processing links from {url}: {e}")

        return rel_path

    except requests.exceptions.RequestException as e:
        error_msg = f"Error downloading {url}: {str(e)}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error saving {url}: {str(e)}"
        logger.error(error_msg)
        return error_msg


def copy_file_url(file_path: str, sandbox_path: str) -> str:
    """
    Copy local file or directory to sandbox (recursive for directories).
    
    Returns:
        Path to copied file/directory relative to sandbox, or error message
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"
        
        # Get the base name
        base_name = os.path.basename(file_path)
        
        # Create 'imported' subdirectory
        import_dir = os.path.join(sandbox_path, 'imported')
        os.makedirs(import_dir, exist_ok=True)
        
        dest_path = os.path.join(import_dir, base_name)
        
        # Handle name conflicts
        if os.path.exists(dest_path):
            name, ext = os.path.splitext(base_name)
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(import_dir, f"{name}_{counter}{ext}")
                counter += 1
        
        # Copy file or directory
        if os.path.isdir(file_path):
            shutil.copytree(file_path, dest_path, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
        else:
            shutil.copy2(file_path, dest_path)
        
        # Return relative path
        rel_path = os.path.relpath(dest_path, sandbox_path)
        logger.info(f"Copied {file_path} to {rel_path}")
        return rel_path
        
    except Exception as e:
        error_msg = f"Error copying {file_path}: {str(e)}"
        logger.error(error_msg)
        return error_msg


def process_urls_in_prompt(prompt: str, sandbox_id: str) -> Tuple[str, List[str]]:
    """
    Process all URLs in a user prompt by downloading/copying them to sandbox.

    Args:
        prompt: User's message text
        sandbox_id: ID of the sandbox

    Returns:
        Tuple of (updated_prompt, list of result messages)
    """
    urls = extract_urls(prompt)
    results = []

    if not urls['web_urls'] and not urls['file_urls']:
        return prompt, []

    sandbox_path = get_sandbox_path(sandbox_id)

    # Check if user requested recursive download
    # Look for patterns like "recursive", "level 1", "depth 1", etc.
    recursive = False
    max_depth = 1

    if re.search(r'\b(recursive|crawl|spider|level|depth)\b', prompt, re.IGNORECASE):
        recursive = True
        # Look for depth/level specification
        depth_match = re.search(r'\b(?:level|depth)\s*(\d+)\b', prompt, re.IGNORECASE)
        if depth_match:
            max_depth = int(depth_match.group(1))

    # Process web URLs
    for url in urls['web_urls']:
        if recursive:
            result = download_web_url(url, sandbox_path, recursive=True, max_depth=max_depth)
            if result.startswith('Error') or result.startswith('Skipped'):
                results.append(f"❌ {result}")
            else:
                results.append(f"✓ Downloaded {url} recursively (depth {max_depth}) to {result}")
        else:
            result = download_web_url(url, sandbox_path, recursive=False)
            if result.startswith('Error'):
                results.append(f"❌ {result}")
            else:
                results.append(f"✓ Downloaded {url} to {result}")

    # Process file URLs
    for file_path in urls['file_urls']:
        result = copy_file_url(file_path, sandbox_path)
        if result.startswith('Error'):
            results.append(f"❌ {result}")
        else:
            results.append(f"✓ Copied {file_path} to {result}")

    # Append results to prompt if any
    if results:
        updated_prompt = prompt + "\n\n---\n**Auto-imported files:**\n" + "\n".join(results)
        return updated_prompt, results

    return prompt, []
