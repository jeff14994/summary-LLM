"""
HTML Extractor Module

This module handles fetching and parsing HTML content from SayIt archive URLs.
"""

import requests
from bs4 import BeautifulSoup
from loguru import logger
from typing import Optional, List

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

    def _extract_speech_content(self, speech_wrapper: BeautifulSoup) -> Optional[str]:
        """
        Extract content from a single speech wrapper div.
        
        Args:
            speech_wrapper (BeautifulSoup): The speech wrapper div element
            
        Returns:
            Optional[str]: The extracted speech content if successful, None otherwise
        """
        try:
            # Extract speaker name
            speaker_elem = speech_wrapper.find('span', class_='speech__meta-data__speaker-name')
            speaker = speaker_elem.get_text(strip=True) if speaker_elem else "Unknown Speaker"
            
            # Extract speech content
            content_elem = speech_wrapper.find('div', class_='speech__content')
            if not content_elem:
                return None
                
            # Get all paragraphs
            paragraphs = content_elem.find_all('p')
            if not paragraphs:
                return None
                
            # Combine speaker and content
            content = "\n".join([f"{speaker}: {p.get_text(strip=True)}" for p in paragraphs])
            return content
            
        except Exception as e:
            logger.error(f"Failed to extract speech content: {str(e)}")
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
            
            # Find all speech wrappers
            speech_wrappers = soup.find_all('div', class_='speech-wrapper')
            if not speech_wrappers:
                logger.warning("No speech content found in the HTML")
                return None
                
            # Extract content from each speech wrapper
            speeches = []
            for wrapper in speech_wrappers:
                speech_content = self._extract_speech_content(wrapper)
                if speech_content:
                    speeches.append(speech_content)
                    
            if not speeches:
                logger.warning("No valid speech content found")
                return None
                
            # Combine all speeches
            return "\n\n".join(speeches)
            
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