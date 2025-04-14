"""
HTML Extractor Module

This module handles fetching and parsing HTML content from SayIt archive URLs.
"""

import requests
from bs4 import BeautifulSoup
from loguru import logger
from typing import Optional

class HTMLExtractor:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def fetch_content(self, url: str) -> Optional[str]:
        """
        Fetch HTML content from the given URL.
        
        Args:
            url (str): The URL to fetch content from
            
        Returns:
            Optional[str]: The HTML content if successful, None otherwise
        """
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch content from {url}: {str(e)}")
            return None

    def extract_transcription(self, html_content: str) -> Optional[str]:
        """
        Extract transcription text from HTML content.
        
        Args:
            html_content (str): The HTML content to parse
            
        Returns:
            Optional[str]: The extracted transcription text if successful, None otherwise
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'footer']):
                element.decompose()
            
            # Extract main content
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            if not main_content:
                logger.warning("No main content found in the HTML")
                return None
                
            # Clean and format text
            text = main_content.get_text(separator='\n', strip=True)
            return text
        except Exception as e:
            logger.error(f"Failed to extract transcription: {str(e)}")
            return None

    def process_url(self, url: str) -> Optional[str]:
        """
        Process a URL to extract transcription content.
        
        Args:
            url (str): The URL to process
            
        Returns:
            Optional[str]: The extracted transcription text if successful, None otherwise
        """
        html_content = self.fetch_content(url)
        if not html_content:
            return None
            
        return self.extract_transcription(html_content) 