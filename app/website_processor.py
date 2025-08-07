import asyncio
import aiohttp
import re
from typing import Optional, Tuple
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import logging
from concurrent.futures import ThreadPoolExecutor
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebsiteProcessor:
    def __init__(self):
        self.session = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Content extraction settings
        self.max_content_length = 50000  # Maximum characters to extract
        self.request_timeout = 30  # Timeout for HTTP requests
        self.max_retries = 3
        
        # Headers to mimic a real browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self.headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def validate_url_accessibility(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if a URL is accessible and returns a valid response.
        Returns (is_accessible, error_message)
        """
        try:
            if not self.session:
                raise ValueError("Session not initialized. Use async context manager.")
            
            logger.info(f"Validating URL accessibility: {url}")
            
            async with self.session.head(url, allow_redirects=True) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '').lower()
                    if 'text/html' in content_type or 'text/plain' in content_type:
                        return True, None
                    else:
                        return False, f"URL does not return HTML content (content-type: {content_type})"
                else:
                    return False, f"URL returned status code: {response.status}"
                    
        except asyncio.TimeoutError:
            return False, "Request timed out"
        except aiohttp.ClientError as e:
            return False, f"Network error: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    async def fetch_website_content(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Fetch and extract content from a website.
        Returns (extracted_text, error_message)
        """
        try:
            if not self.session:
                raise ValueError("Session not initialized. Use async context manager.")
            
            logger.info(f"Fetching website content from: {url}")
            start_time = time.time()
            
            # First validate the URL
            is_accessible, error_msg = await self.validate_url_accessibility(url)
            if not is_accessible:
                return None, error_msg
            
            # Fetch the actual content
            async with self.session.get(url, allow_redirects=True) as response:
                if response.status != 200:
                    return None, f"Failed to fetch content: HTTP {response.status}"
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type and 'text/plain' not in content_type:
                    return None, f"Unsupported content type: {content_type}"
                
                # Read content with size limit
                content = await response.text()
                
                if len(content) > self.max_content_length * 2:  # Allow some buffer for processing
                    logger.warning(f"Content too large ({len(content)} chars), truncating")
                    content = content[:self.max_content_length * 2]
                
                # Extract text content
                extracted_text = await self._extract_text_content(content, url)
                
                fetch_time = time.time() - start_time
                logger.info(f"Successfully extracted {len(extracted_text)} characters in {fetch_time:.2f}s")
                
                return extracted_text, None
                
        except asyncio.TimeoutError:
            return None, "Request timed out while fetching content"
        except aiohttp.ClientError as e:
            return None, f"Network error while fetching content: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error fetching website content: {str(e)}")
            return None, f"Unexpected error: {str(e)}"
    
    async def _extract_text_content(self, html_content: str, base_url: str) -> str:
        """
        Extract meaningful text content from HTML, filtering out navigation, ads, etc.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._extract_text_sync, html_content, base_url)
    
    def _extract_text_sync(self, html_content: str, base_url: str) -> str:
        """
        Synchronous text extraction from HTML content.
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            unwanted_tags = [
                'script', 'style', 'nav', 'header', 'footer', 'aside',
                'advertisement', 'ads', 'sidebar', 'menu', 'breadcrumb',
                'social', 'share', 'comment', 'popup', 'modal'
            ]
            
            for tag in unwanted_tags:
                for element in soup.find_all(tag):
                    element.decompose()
            
            # Remove elements with common ad/navigation class names and IDs
            unwanted_patterns = [
                'nav', 'menu', 'sidebar', 'footer', 'header', 'ad', 'advertisement',
                'social', 'share', 'comment', 'popup', 'modal', 'cookie', 'banner'
            ]
            
            for pattern in unwanted_patterns:
                # Remove by class
                for element in soup.find_all(class_=re.compile(pattern, re.I)):
                    element.decompose()
                # Remove by ID
                for element in soup.find_all(id=re.compile(pattern, re.I)):
                    element.decompose()
            
            # Focus on main content areas
            main_content_selectors = [
                'main', 'article', '[role="main"]', '.main-content', 
                '.content', '.post-content', '.entry-content', '.page-content'
            ]
            
            main_content = None
            for selector in main_content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            # If no main content found, use the body
            if not main_content:
                main_content = soup.find('body') or soup
            
            # Extract text from paragraphs, headings, and lists
            text_elements = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div'])
            
            extracted_texts = []
            for element in text_elements:
                text = element.get_text(strip=True)
                if text and len(text) > 20:  # Filter out very short text snippets
                    extracted_texts.append(text)
            
            # Join all text with proper spacing
            full_text = '\n\n'.join(extracted_texts)
            
            # Clean up the text
            full_text = self._clean_extracted_text(full_text)
            
            # Truncate if too long
            if len(full_text) > self.max_content_length:
                full_text = full_text[:self.max_content_length] + "..."
                logger.info(f"Content truncated to {self.max_content_length} characters")
            
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text content: {str(e)}")
            return f"Error extracting content from website: {str(e)}"
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean and normalize extracted text content.
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove common unwanted phrases
        unwanted_phrases = [
            'click here', 'read more', 'learn more', 'subscribe now',
            'sign up', 'log in', 'register', 'download now', 'buy now',
            'add to cart', 'share this', 'follow us', 'like us'
        ]
        
        for phrase in unwanted_phrases:
            text = re.sub(re.escape(phrase), '', text, flags=re.IGNORECASE)
        
        # Clean up any remaining artifacts
        text = text.strip()
        
        return text

    async def process_website_url(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Main method to process a website URL and extract content.
        Returns (extracted_content, error_message)
        """
        if not url or not url.strip():
            return None, "No URL provided"
        
        try:
            async with self:  # Use context manager
                content, error = await self.fetch_website_content(url.strip())
                return content, error
        except Exception as e:
            logger.error(f"Error processing website URL {url}: {str(e)}")
            return None, f"Error processing website: {str(e)}"
