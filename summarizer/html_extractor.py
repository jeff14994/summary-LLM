"""
HTML Extractor Module

This module handles fetching and parsing HTML content from various URLs.
"""

import requests
from bs4 import BeautifulSoup
from loguru import logger
from typing import Optional, List, Dict

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

    def _extract_paragraphs(self, element: BeautifulSoup) -> List[str]:
        """
        Extract paragraphs from any HTML element.
        
        Args:
            element (BeautifulSoup): The HTML element to extract from
            
        Returns:
            List[str]: List of extracted paragraph texts
        """
        paragraphs = []
        # Find all p tags
        for p in element.find_all('p'):
            text = p.get_text(strip=True)
            if text:  # Only add non-empty paragraphs
                paragraphs.append(text)
        return paragraphs

    def _extract_speakers(self, element: BeautifulSoup) -> Dict[str, List[str]]:
        """
        Extract speakers and their content from HTML.
        Tries multiple common patterns for speaker identification.
        
        Args:
            element (BeautifulSoup): The HTML element to extract from
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping speakers to their paragraphs
        """
        speakers = {}
        current_speaker = "Unknown Speaker"
        
        # Try different patterns for speaker identification
        speaker_patterns = [
            ('span', {'class': 'speaker'}),  # Common class for speakers
            ('strong', {}),                  # Speakers often in strong tags
            ('b', {}),                       # Or bold tags
            ('div', {'class': 'speaker'}),   # Div with speaker class
            ('h3', {}),                      # Sometimes in headings
            ('h4', {})
        ]
        
        # Find all potential speaker elements
        for tag, attrs in speaker_patterns:
            for speaker_elem in element.find_all(tag, attrs):
                speaker_text = speaker_elem.get_text(strip=True)
                if speaker_text:
                    current_speaker = speaker_text
                    if current_speaker not in speakers:
                        speakers[current_speaker] = []
        
        return speakers

    def extract_content(self, html_content: str) -> Optional[str]:
        """
        Extract content from HTML, trying multiple approaches.
        
        Args:
            html_content (str): The HTML content to parse
            
        Returns:
            Optional[str]: The extracted content if successful, None otherwise
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Try different content containers
            content_containers = [
                soup.find('div', {'class': 'content'}),      # Common content class
                soup.find('div', {'class': 'article'}),      # Article content
                soup.find('div', {'class': 'main'}),         # Main content
                soup.find('article'),                        # HTML5 article tag
                soup.find('main'),                           # HTML5 main tag
                soup.find('div', {'id': 'content'}),         # Content by ID
                soup.find('div', {'id': 'main'})             # Main by ID
            ]
            
            # Get the first non-None container
            container = next((c for c in content_containers if c is not None), None)
            if not container:
                logger.warning("No content container found, trying body")
                container = soup.body
                
            if not container:
                logger.error("No content found in HTML")
                return None
            
            # Extract speakers and their content
            speakers = self._extract_speakers(container)
            
            # If we found speakers, format content with speakers
            if speakers:
                content_parts = []
                for speaker, paragraphs in speakers.items():
                    if paragraphs:
                        content_parts.append(f"{speaker}: {' '.join(paragraphs)}")
                return "\n\n".join(content_parts)
            
            # If no speakers found, just extract all paragraphs
            paragraphs = self._extract_paragraphs(container)
            if paragraphs:
                return "\n\n".join(paragraphs)
            
            logger.warning("No content found in the container")
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract content: {str(e)}")
            return None

    def process_url(self, url: str) -> Optional[str]:
        """
        Process a URL to extract content.
        
        Args:
            url (str): The URL to process
            
        Returns:
            Optional[str]: The extracted content if successful, None otherwise
        """
        html_content = self.fetch_content(url)
        if not html_content:
            return None
            
        return self.extract_content(html_content) 